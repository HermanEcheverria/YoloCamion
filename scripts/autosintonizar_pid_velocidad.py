"""Auto-sintonizador PID de velocidad para ETS2.

Ajusta Kp/Ki/Kd de _CFG_VELOCIDAD_DEFAULT automaticamente usando descenso
por coordenadas sobre el MSE del error de velocidad (setpoint vs lectura OCR).

Uso:
    python scripts/autosintonizar_pid_velocidad.py --vel-target 60 --delay 15
    python scripts/autosintonizar_pid_velocidad.py --kp 0.65 --ki 0.05 --kd 0.04 --vel-target 60
    python scripts/autosintonizar_pid_velocidad.py --dry-run

Recomendaciones antes de correr:
  - Pon el camion en una autopista abierta a ~vel-target km/h (cruise control ETS2).
  - Usa --delay 15 para tener tiempo de posicionarte.
  - El script imprime progreso en consola; no necesitas cambiar de ventana.
  - Frames donde el OCR no puede leer la velocidad se ignoran automaticamente.

Al terminar imprime la linea exacta para pegar en gamepad_pid.py.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.percepcion.velocidad_dashboard import EstimadorVelocidadDashboard
from src.fuente.pantalla import FuentePantalla
from src.control.pid import PIDController

# ── Espacio de busqueda ────────────────────────────────────────────────────────
_NOMBRES  = ("Kp",   "Ki",    "Kd")
_INIT     = (0.65,   0.050,   0.040)
_MIN      = (0.10,   0.000,   0.000)
_MAX      = (3.00,   0.500,   0.200)
_STEP     = (0.20,   0.020,   0.020)
_STEP_MIN = (0.02,   0.002,   0.002)
_DECAY    = 0.65

_EVAL_FRAMES   = 90      # frames validos (con OCR) por ventana de evaluacion
_MEJORA_MIN    = 0.02    # mejora relativa minima para aceptar cambio (2%)
_MAX_KMH       = 90.0    # km/h que equivale a vel_norm=1.0
_VEL_MIN_KMH   = 5.0     # debajo de esto el camion esta detenido; frame invalido
_FRENO_EMERG   = 10.0    # si vel supera target + esto km/h, frenar de emergencia
_VEL_MIN_FRENO = 0.03    # no aplicar LT si vel_norm < esto (evita reversa ETS2)
_VEL_ARRANQUE  = 0.06    # ~5 km/h: gas completo hasta superar esto

_DT_LOOP = 0.08          # ~12 Hz (suficiente para control de velocidad)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── Motor de autosintonizacion ─────────────────────────────────────────────────

class AutosintonizadorVel:
    """Descenso por coordenadas para gains de PID de velocidad.

    Acepta errores de velocidad normalizados uno a uno via procesar().
    valido=False descarta el frame del MSE (ej: OCR fallido o camion parado).
    """

    def __init__(self, gains_init, steps, steps_min, step_decay,
                 eval_frames, mejora_min, gains_min, gains_max):
        self.gains        = list(gains_init)
        self.best_gains   = list(gains_init)
        self.best_mse     = float("inf")
        self.steps        = list(steps)
        self.steps_min    = list(steps_min)
        self.decay        = step_decay
        self.eval_frames  = eval_frames
        self.mejora_min   = mejora_min
        self.gains_min    = list(gains_min)
        self.gains_max    = list(gains_max)

        self.param_idx    = 0
        self.fase         = "baseline"
        self.frames_fase  = 0
        self.errs_fase: list[float] = []
        self.baseline_mse = float("inf")
        self.gains_prueba = list(gains_init)
        self.converged    = False
        self.epochs       = 0

    @property
    def gains_a_aplicar(self) -> list:
        return self.gains_prueba[:]

    def procesar(self, error_norm: float, valido: bool = True) -> dict | None:
        if not valido:
            return None
        self.errs_fase.append(error_norm)
        self.frames_fase += 1
        if self.frames_fase >= self.eval_frames:
            return self._evaluar()
        return None

    def _mse(self) -> float:
        return float(np.mean(np.array(self.errs_fase) ** 2)) if self.errs_fase else 0.0

    def _reset_fase(self) -> None:
        self.frames_fase = 0
        self.errs_fase   = []

    def _evaluar(self) -> dict:
        mse = self._mse()
        self._reset_fase()

        if self.fase == "baseline":
            self.baseline_mse = mse
            if mse < self.best_mse:
                self.best_mse   = mse
                self.best_gains = self.gains[:]
            return self._intentar("plus")

        if self.fase == "plus":
            if mse < self.baseline_mse * (1.0 - self.mejora_min):
                return self._aceptar(mse, "+")
            return self._intentar("minus")

        if mse < self.baseline_mse * (1.0 - self.mejora_min):
            return self._aceptar(mse, "-")
        return self._rechazar(mse)

    def _intentar(self, direction: str) -> dict:
        p    = self.param_idx
        sign = +1.0 if direction == "plus" else -1.0
        val  = self.gains[p] + sign * self.steps[p]
        val  = float(np.clip(val, self.gains_min[p], self.gains_max[p]))

        if abs(val - self.gains[p]) < 1e-9:
            if direction == "plus":
                return self._intentar("minus")
            return self._rechazar(self.baseline_mse)

        self.gains_prueba    = self.gains[:]
        self.gains_prueba[p] = round(val, 5)
        self.fase            = direction
        return {
            "tipo": "nuevo_test",
            "direction": direction,
            "param": _NOMBRES[p],
            "gains_prueba": self.gains_prueba[:],
            "baseline_mse": round(self.baseline_mse, 5),
            "step": self.steps[p],
        }

    def _aceptar(self, mse: float, direction: str) -> dict:
        p = self.param_idx
        self.gains = self.gains_prueba[:]
        if mse < self.best_mse:
            self.best_mse   = mse
            self.best_gains = self.gains[:]
        ev = {
            "tipo": "aceptado",
            "direction": direction,
            "param": _NOMBRES[p],
            "mse_nuevo": round(mse, 5),
            "mse_base":  round(self.baseline_mse, 5),
            "gains": self.gains[:],
        }
        self._avanzar()
        return ev

    def _rechazar(self, mse_trial: float) -> dict:
        p = self.param_idx
        self.gains_prueba = self.gains[:]
        self.steps[p] = max(self.steps[p] * self.decay, self.steps_min[p])
        ev = {
            "tipo": "rechazado",
            "param": _NOMBRES[p],
            "mse_base":   round(self.baseline_mse, 5),
            "mse_trial":  round(mse_trial, 5),
            "nuevo_step": round(self.steps[p], 5),
            "gains": self.gains[:],
        }
        self._avanzar()
        return ev

    def _avanzar(self) -> None:
        self.param_idx = (self.param_idx + 1) % 3
        self.fase      = "baseline"
        if self.param_idx == 0:
            self.epochs += 1
        if all(s <= sm + 1e-9 for s, sm in zip(self.steps, self.steps_min)):
            self.converged = True


# ── Impresion de eventos ───────────────────────────────────────────────────────

def _imprimir_evento(ev: dict) -> None:
    t    = _ts()
    tipo = ev["tipo"]

    if tipo == "nuevo_test":
        g = ev["gains_prueba"]
        d = "+" if ev["direction"] == "plus" else "-"
        print(f"\n[{t}] >> PROBAR {ev['param']}{d}{ev['step']:.4g} "
              f"| Kp={g[0]:.3f} Ki={g[1]:.4f} Kd={g[2]:.4f} "
              f"| MSE_base={ev['baseline_mse']:.5f}")

    elif tipo == "aceptado":
        g = ev["gains"]
        mejora = (ev["mse_base"] - ev["mse_nuevo"]) / max(ev["mse_base"], 1e-9) * 100
        print(f"[{t}] OK {ev['param']}{ev['direction']} "
              f"| {ev['mse_base']:.5f} -> {ev['mse_nuevo']:.5f} (-{mejora:.1f}%) "
              f"| Kp={g[0]:.3f} Ki={g[1]:.4f} Kd={g[2]:.4f}")

    elif tipo == "rechazado":
        g = ev["gains"]
        print(f"[{t}] NO {ev['param']:2s} "
              f"| base={ev['mse_base']:.5f} trial={ev['mse_trial']:.5f} "
              f"| step->{ev['nuevo_step']:.4g} "
              f"| Kp={g[0]:.3f} Ki={g[1]:.4f} Kd={g[2]:.4f}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vel-target", type=float, default=60.0,
                        help="Velocidad objetivo en km/h (default 60)")
    parser.add_argument("--max-kmh",   type=float, default=_MAX_KMH,
                        help="km/h que equivale a vel_norm=1.0 (default 90)")
    parser.add_argument("--delay",     type=int,   default=10,
                        help="Segundos de countdown (default 10)")
    parser.add_argument("--kp",        type=float, default=_INIT[0])
    parser.add_argument("--ki",        type=float, default=_INIT[1])
    parser.add_argument("--kd",        type=float, default=_INIT[2])
    parser.add_argument("--eval-frames", type=int, default=_EVAL_FRAMES,
                        help=f"Frames validos por ventana (default {_EVAL_FRAMES})")
    parser.add_argument("--max-min",   type=int, default=30,
                        help="Tiempo maximo en minutos (default 30)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Sin gamepad (solo correr el algoritmo)")
    args = parser.parse_args()

    max_kmh     = float(args.max_kmh)
    target_kmh  = float(args.vel_target)
    target_norm = min(1.0, target_kmh / max_kmh)
    t_limite    = time.monotonic() + args.max_min * 60
    emerg_kmh   = target_kmh + _FRENO_EMERG

    # ── Countdown ──────────────────────────────────────────────────────────────
    if args.delay > 0:
        print(f"\nPon el camion a ~{target_kmh:.0f} km/h en una autopista abierta.")
        print(f"Arrancando en {args.delay}s...")
        for i in range(args.delay, 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1)
        print("  Iniciando autosintonizador!\n")

    # ── Componentes ────────────────────────────────────────────────────────────
    estimador = EstimadorVelocidadDashboard(max_kmh_norm=max_kmh, retener_frames=8)
    fuente    = FuentePantalla(monitor=0, escalar_a=(1920, 1080))
    fuente.iniciar()

    pid = PIDController(kp=args.kp, ki=args.ki, kd=args.kd, limite=1.0)

    gamepad = None
    if not args.dry_run:
        try:
            import vgamepad as vg
            gamepad = vg.VX360Gamepad()
            print(f"[{_ts()}] Gamepad virtual conectado. Objetivo={target_kmh:.0f} km/h")
        except Exception as e:
            print(f"[{_ts()}] [!] Gamepad no disponible: {e}. Modo dry-run.")

    auto = AutosintonizadorVel(
        gains_init  = (args.kp, args.ki, args.kd),
        steps       = list(_STEP),
        steps_min   = list(_STEP_MIN),
        step_decay  = _DECAY,
        eval_frames = args.eval_frames,
        mejora_min  = _MEJORA_MIN,
        gains_min   = list(_MIN),
        gains_max   = list(_MAX),
    )

    gains_pid_actual = auto.gains_a_aplicar

    print(f"[{_ts()}] === Autosintonizador de VELOCIDAD iniciado ===")
    print(f"         Gains iniciales : Kp={args.kp:.3f}  Ki={args.ki:.4f}  Kd={args.kd:.4f}")
    print(f"         Objetivo        : {target_kmh:.0f} km/h  (norm={target_norm:.3f})")
    print(f"         Eval frames     : {args.eval_frames}  (solo frames con OCR valido)")
    print(f"         Mejora minima   : {_MEJORA_MIN*100:.0f}%")
    print(f"         Tiempo maximo   : {args.max_min} min")
    print(f"         Parar con Ctrl+C para ver resultado parcial.\n")

    # ── Loop principal ─────────────────────────────────────────────────────────
    t_prev     = time.monotonic()
    n_frames   = 0
    n_ocr_fail = 0

    try:
        while not auto.converged:
            if time.monotonic() > t_limite:
                print(f"\n[{_ts()}] Tiempo maximo alcanzado ({args.max_min} min).")
                break

            t_now  = time.monotonic()
            dt     = max(0.010, t_now - t_prev)
            t_prev = t_now

            cuadro = fuente.siguiente()
            if cuadro is None:
                time.sleep(0.01)
                continue

            lectura  = estimador.estimar(cuadro.imagen)
            vel_kmh  = lectura.kmh   # None si OCR fallo
            vel_norm = lectura.norm  # 0.0 si None

            ocr_ok = vel_kmh is not None and vel_kmh >= _VEL_MIN_KMH

            # Actualizar PID si cambiaron los gains
            nuevos = auto.gains_a_aplicar
            if nuevos != gains_pid_actual:
                pid._kp, pid._ki, pid._kd = nuevos
                pid.reset()
                gains_pid_actual = nuevos

            # Control: arranque o PID normal
            if vel_norm < _VEL_ARRANQUE:
                rt_aplicado = 255
                lt_aplicado = 0
                pid.reset()
            else:
                pid_out = pid.calcular(target_norm, vel_norm, dt)
                if pid_out >= 0.0:
                    rt_aplicado = int(min(1.0, pid_out) * 255)
                    lt_aplicado = 0
                else:
                    rt_aplicado = 0
                    if vel_norm >= _VEL_MIN_FRENO:
                        lt_aplicado = int(min(1.0, -pid_out) * 255)
                    else:
                        lt_aplicado = 0

            # Seguridad: freno de emergencia si velocidad se dispara
            if vel_kmh is not None and vel_kmh > emerg_kmh:
                rt_aplicado = 0
                lt_aplicado = min(255, int((vel_kmh - emerg_kmh) / 10.0 * 255))

            if gamepad is not None:
                gamepad.right_trigger(value=rt_aplicado)
                gamepad.left_trigger(value=lt_aplicado)
                gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
                gamepad.update()

            if not ocr_ok:
                n_ocr_fail += 1

            error_norm = target_norm - vel_norm
            evento = auto.procesar(error_norm, valido=ocr_ok)

            if evento is not None:
                _imprimir_evento(evento)
                if evento["tipo"] == "nuevo_test":
                    pid.reset()

            n_frames += 1

            if n_frames % 20 == 0:
                errs_rec = auto.errs_fase[-20:] if len(auto.errs_fase) >= 5 else auto.errs_fase
                mse_p    = float(np.mean(np.array(errs_rec) ** 2)) if errs_rec else 0.0
                g        = auto.gains_a_aplicar
                ocr_pct  = int(100 * n_ocr_fail / n_frames)
                vel_str  = f"{vel_kmh:.1f}" if vel_kmh is not None else "----"
                print(f"[{_ts()}] {auto.fase:8s} {_NOMBRES[auto.param_idx]:2s} "
                      f"| validos {auto.frames_fase:3d}/{auto.eval_frames} ocr_fail={ocr_pct:2d}% "
                      f"| MSE={mse_p:.5f} vel={vel_str:5s} RT={rt_aplicado:3d} LT={lt_aplicado:3d} "
                      f"| Kp={g[0]:.3f} Ki={g[1]:.4f} Kd={g[2]:.4f}"
                      f"{'  [MEJOR='+f'{auto.best_mse:.5f}]' if auto.best_mse < float('inf') else ''}")

            elapsed  = time.monotonic() - t_now
            restante = _DT_LOOP - elapsed
            if restante > 0:
                time.sleep(restante)

    except KeyboardInterrupt:
        print(f"\n[{_ts()}] Interrumpido por Ctrl+C.")
    finally:
        if gamepad is not None:
            gamepad.right_trigger(value=0)
            gamepad.left_trigger(value=0)
            gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            gamepad.update()
        fuente.cerrar()

    # ── Resumen final ──────────────────────────────────────────────────────────
    ocr_pct = int(100 * n_ocr_fail / n_frames) if n_frames else 0
    cg = auto.gains
    print(f"\n[{_ts()}] === Autosintonizador terminado ===")
    print(f"         Frames procesados : {n_frames}  (OCR fallido ignorado: {n_ocr_fail} = {ocr_pct}%)")
    print(f"         Epocas completadas: {auto.epochs}")
    print(f"         Convergido        : {auto.converged}")

    if auto.best_mse < float("inf"):
        bg = auto.best_gains
        print(f"\n  Gains aceptados (usar estos):")
        print(f"    Kp={cg[0]:.3f}  Ki={cg[1]:.4f}  Kd={cg[2]:.4f}")
        if bg != cg:
            print(f"  (Mejor MSE={auto.best_mse:.5f} fue con Kp={bg[0]:.3f} Ki={bg[1]:.4f} Kd={bg[2]:.4f}, "
                  f"puede ser de un tramo mas estable)")
        print()
        print("  Pega esta linea en src/control/gamepad_pid.py:")
        print(f"    _CFG_VELOCIDAD_DEFAULT = ConfigPID("
              f"kp={cg[0]:.3f}, ki={cg[1]:.4f}, kd={cg[2]:.4f})")
    else:
        print("         No se completo ninguna evaluacion.")
        print("         Asegurate que el camion este en movimiento y el OCR del HUD funcione.")

    if auto.converged:
        print("\n  El algoritmo convergio. Los gains son optimos dentro del espacio de busqueda.")
    else:
        start = cg if auto.best_mse < float("inf") else (args.kp, args.ki, args.kd)
        print(f"\n  Puedes continuar con:")
        print(f"    python scripts/autosintonizar_pid_velocidad.py"
              f" --kp {start[0]:.3f} --ki {start[1]:.4f} --kd {start[2]:.4f}"
              f" --vel-target {target_kmh:.0f} --delay 10")


if __name__ == "__main__":
    main()
