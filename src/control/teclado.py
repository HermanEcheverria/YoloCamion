"""Controlador de teclado con PWM para steering suave (DEPRECATED).

DEPRECATED desde Tarea 3.5 (refactor pure-vision):
    El control oficial es `ControladorGamepadPID` (gamepad analogico con
    tres PIDs alimentados por velocidad propia visual). Este controlador
    de teclado se mantiene SOLO como fallback de emergencia. No recibe
    desarrollo nuevo y no se beneficia de los PIDs ni del setpoint
    continuo del FSM.

W/S se mantienen pulsadas mientras la condición persiste.
A/D usan pulsos cortos proporcionales a la magnitud de desviación,
lo que simula control analógico con teclas binarias.
"""
import logging
import time

try:
    import pydirectinput
except ImportError:
    pydirectinput = None  # solo disponible en Windows

from src.tipos import ComandoControl
from src.control.base import Controlador

logger = logging.getLogger(__name__)

_UMBRAL_AVANZAR = 0.20   # MANTENER(0.3) y ACELERAR(0.6) presionan W
_UMBRAL_FRENAR  = 0.15   # cualquier freno real activa S
_UMBRAL_GIRAR   = 0.25   # mínima desviación para activar A/D

# Duración del pulso de steering (ms) según magnitud de desviación
# Desviación 0.25 → 12ms | 0.5 → 20ms | 1.0 → 40ms
_MS_PULSO_MIN = 12
_MS_PULSO_MAX = 40


def _duracion_pulso(magnitud: float) -> float:
    """Duración en segundos del pulso A/D proporcional a la desviación."""
    t = (magnitud - _UMBRAL_GIRAR) / (1.0 - _UMBRAL_GIRAR)
    ms = _MS_PULSO_MIN + t * (_MS_PULSO_MAX - _MS_PULSO_MIN)
    return max(_MS_PULSO_MIN, min(_MS_PULSO_MAX, ms)) / 1000.0


class ControladorTeclado(Controlador):
    """Teclado con PWM para steering proporcional.

    Requiere ejecutar como Administrador cuando ETS2 corre con privilegios elevados.
    """

    def __init__(self):
        logger.warning(
            "ControladorTeclado esta DEPRECATED desde Tarea 3.5; "
            "usar ControladorGamepadPID. Solo se mantiene como fallback."
        )
        self._teclas_continuas: set[str] = set()   # W, S — se mantienen
        self._steering_activo: str | None = None    # última tecla de giro

    def aplicar(self, cmd: ComandoControl) -> None:
        # ── Velocidad: W/S se mantienen mientras la condición persiste ───────
        deseadas: set[str] = set()
        if cmd.acelerador >= _UMBRAL_AVANZAR:
            deseadas.add("w")
        if cmd.freno >= _UMBRAL_FRENAR:
            deseadas.add("s")
        if "w" in deseadas and "s" in deseadas:
            deseadas.discard("w")  # freno tiene prioridad

        for tecla in self._teclas_continuas - deseadas:
            pydirectinput.keyUp(tecla)
        for tecla in deseadas - self._teclas_continuas:
            pydirectinput.keyDown(tecla)
        self._teclas_continuas = deseadas

        # ── Steering: siempre soltar antes de aplicar nuevo pulso ────────────
        if self._steering_activo:
            pydirectinput.keyUp(self._steering_activo)
            self._steering_activo = None

        volante = cmd.volante
        if abs(volante) >= _UMBRAL_GIRAR:
            tecla = "a" if volante < 0 else "d"
            duracion = _duracion_pulso(abs(volante))
            pydirectinput.keyDown(tecla)
            time.sleep(duracion)
            pydirectinput.keyUp(tecla)
            # No guardamos como activo — siempre se libera dentro del mismo ciclo

    def liberar(self) -> None:
        for tecla in list(self._teclas_continuas):
            pydirectinput.keyUp(tecla)
        self._teclas_continuas.clear()
        if self._steering_activo:
            pydirectinput.keyUp(self._steering_activo)
            self._steering_activo = None
        logger.info("Teclado liberado")

    def cerrar(self) -> None:
        self.liberar()
