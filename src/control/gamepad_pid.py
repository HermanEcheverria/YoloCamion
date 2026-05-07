"""Controlador de gamepad 100% analogico — acelerador open-loop.

Volante: PID sobre error_carril (PurePursuit) o stick directo (FSM).
Velocidad: RT = velocidad_objetivo_norm * 255 directo — sin PID, sin OCR.
Frenado: LT = freno_objetivo * 255 directo.

Anti-reversa: si llevamos mas de _FRAMES_PARADO_EST frames consecutivos
frenando (RT=0, LT>0), asumimos que el camion ya paro y bloqueamos LT
para que ETS2 no engrane reversa.
"""
import logging
from dataclasses import dataclass

from src.control.base import Controlador
from src.control.pid import PIDController
from src.tipos import ComandoControl, SetpointControl

logger = logging.getLogger(__name__)


@dataclass
class ConfigPID:
    kp: float
    ki: float
    kd: float


_CFG_VOLANTE_DEFAULT = ConfigPID(kp=0.220, ki=0.0050, kd=0.330)

_FRENO_DIRECTO_MIN = 0.05   # freno_objetivo >= esto activa LT
_FRAMES_PARADO_EST = 900    # ~56s a 16fps frenando → asumir camion parado → bloquear LT
                            # (con LT=freno el juego no engrana reversa; el limite es
                            # solo seguridad para detenciones muy largas)


class ControladorGamepadPID(Controlador):
    """Gamepad Xbox virtual (vgamepad): volante PID, velocidad open-loop."""

    def __init__(
        self,
        cfg_volante: ConfigPID = _CFG_VOLANTE_DEFAULT,
        cfg_velocidad: ConfigPID | None = None,   # ignorado, conservado por compatibilidad
    ):
        self._pid_vol = PIDController(
            cfg_volante.kp, cfg_volante.ki, cfg_volante.kd, limite=1.0
        )
        self._gamepad = None
        self._t_ultimo: float | None = None
        self._frames_frenando: int = 0
        self._ultimo_rt = 0
        self._ultimo_lt = 0
        self._ultimo_stick = 0.0

    def iniciar(self) -> None:
        import vgamepad as vg
        self._gamepad = vg.VX360Gamepad()
        logger.info("ControladorGamepadPID: gamepad virtual iniciado")

    def actualizar_velocidad_actual(self, velocidad_norm: float) -> None:
        """Conservado por compatibilidad con el bucle del piloto; sin efecto."""

    # ── Compatibilidad con la API vieja (acepta ComandoControl) ─────────────
    def aplicar(self, sp_o_cmd) -> None:
        if isinstance(sp_o_cmd, ComandoControl):
            sp = SetpointControl(
                velocidad_objetivo_norm=sp_o_cmd.acelerador,
                freno_objetivo=sp_o_cmd.freno,
                desviacion_volante=sp_o_cmd.volante,
            )
        elif isinstance(sp_o_cmd, SetpointControl):
            sp = sp_o_cmd
        else:
            raise TypeError(
                f"aplicar() espera SetpointControl o ComandoControl, "
                f"recibio {type(sp_o_cmd).__name__}"
            )
        self._aplicar_setpoint(sp)

    def _aplicar_setpoint(self, sp: SetpointControl) -> None:
        if self._gamepad is None:
            raise RuntimeError("Llamar a iniciar() antes de aplicar()")

        import time as _t
        ahora = _t.monotonic()
        dt = 0.033 if self._t_ultimo is None else max(0.001, ahora - self._t_ultimo)
        self._t_ultimo = ahora

        # ── Volante ──────────────────────────────────────────────────────────
        if sp.error_carril is not None:
            stick_x = max(-1.0, min(1.0,
                self._pid_vol.calcular(0.0, sp.error_carril, dt)
            ))
        else:
            stick_x = max(-1.0, min(1.0, float(sp.desviacion_volante)))

        self._gamepad.left_joystick_float(
            x_value_float=float(stick_x), y_value_float=0.0
        )

        # ── Velocidad / Frenado (open-loop) ──────────────────────────────────
        if sp.freno_objetivo >= _FRENO_DIRECTO_MIN:
            rt_aplicado = 0
            self._frames_frenando += 1
            # Tras _FRAMES_PARADO_EST frames continuos frenando se asume paro:
            # bloqueamos LT para que ETS2 no engrane reversa.
            if self._frames_frenando < _FRAMES_PARADO_EST:
                lt_aplicado = int(min(1.0, sp.freno_objetivo) * 255)
            else:
                lt_aplicado = 0
        else:
            rt_aplicado = int(min(1.0, sp.velocidad_objetivo_norm) * 255)
            lt_aplicado = 0
            self._frames_frenando = 0

        self._gamepad.right_trigger(value=rt_aplicado)
        self._gamepad.left_trigger(value=lt_aplicado)
        self._ultimo_rt = rt_aplicado
        self._ultimo_lt = lt_aplicado
        self._ultimo_stick = float(stick_x)

        self._gamepad.update()

    @property
    def ultimo_comando_aplicado(self) -> tuple[int, int, float]:
        """(rt, lt, stick_x) del ultimo aplicar(); util para debug-piloto."""
        return self._ultimo_rt, self._ultimo_lt, self._ultimo_stick

    def liberar(self) -> None:
        if self._gamepad is None:
            return
        self._gamepad.right_trigger(value=0)
        self._gamepad.left_trigger(value=0)
        self._gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self._gamepad.update()
        self._pid_vol.reset()
        self._frames_frenando = 0
        logger.info("ControladorGamepadPID: ejes liberados")

    def cerrar(self) -> None:
        self.liberar()
