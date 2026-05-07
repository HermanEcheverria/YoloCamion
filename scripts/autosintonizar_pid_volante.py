"""Auto-sintonizador PID de volante para ETS2.

No requiere interaccion del usuario mientras corre. Ajusta Kp/Ki/Kd
automaticamente usando descenso por coordenadas:

    Para cada parametro (Kp, Ki, Kd):
      1. Evalua MSE con los gains actuales (EVAL_FRAMES frames)
      2. Prueba param + delta  → si MSE mejora >= MEJORA_MIN: guarda
      3. Si no mejora: prueba param - delta → si mejora: guarda
      4. Si ninguno mejora: reduce delta (convergencia)
    Repite hasta que todos los deltas esten en el minimo.

Uso:
    python scripts/autosintonizar_pid_volante.py --delay 15 --rt 80
    python scripts/autosintonizar_pid_volante.py --kp 0.22 --ki 0.005 --kd 0.28 --rt 80 --delay 10
    python scripts/autosintonizar_pid_volante.py --dry-run     # sin gamepad (prueba el algoritmo)

Recomendaciones antes de correr:
  - Pon el camion en una carretera recta a 40-80 km/h con cruise control.
  - Usa --delay 15 para tener tiempo de posicionarte.
  - Con --rt 80 el script aplica gas fijo; omitelo si usas cruise control.
  - El script imprime progreso en consola; no necesitas cambiar de ventana.

Al terminar imprime la linea exacta para pegar en gamepad_pid.py.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fuente.pantalla import FuentePantalla
from src.control.pid import PIDController
from src.percepcion.yolop_inference import InferenciaYOLOP
from src.control.pure_pursuit import PurePursuitVisual

# ── Espacio de busqueda ────────────────────────────────────────────────────────
_NOMBRES   = ("Kp",  "Ki",   "Kd")
_INIT      = (0.22,  0.005,  0.28)   # punto de partida sugerido
_MIN       = (0.05,  0.000,  0.000)
_MAX       = (2.00,  0.100,  0.800)
_STEP      = (0.10,  0.005,  0.050)  # delta inicial por parametro
_STEP_MIN  = (0.01,  0.001,  0.005)  # delta minimo = criterio de convergencia
_DECAY     = 0.65   # factor de reduccion de delta cuando no hay mejora

_EVAL_FRAMES    = 90    # frames por ventana de evaluacion (~6-8s a 12fps)
_MEJORA_MIN     = 0.02  # mejora minima relativa para aceptar cambio (2%)
_MSE_EMERGENCIA = 0.30  # MSE parcial maximo antes de revertir en emergencia
_VENTANA_EMERGENCIA = 15  # frames recientes para detectar emergencia

# Parametros de percepcion (igual que ejecutar_piloto.py)
_ROI_FRAC    = 0.60
_ALPHA_EMA   = 0.45
_ZONA_MUERTA = 0.015
_YOLOP_CADA  = 2
_DT_LOOP     = 0.033
_IMGSZ       = 864
_DEVICE      = "cuda"


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _aplicar_gamepad(gamepad, stick: float, rt: int) -> None:
    gamepad.left_joystick_float(x_value_float=float(stick), y_value_float=0.0)
    gamepad.right_trigger(value=rt)
    gamepad.left_trigger(value=0)
    gamepad.update()


# ── Motor de autosintonizacion ─────────────────────────────────────────────────

class Autosintonizador:
    """Descenso por coordenadas para gains de PID de volante.

    Cada llamada a procesar(error) ingiere un error de carril.
    Cuando completa una ventana de evaluacion retorna un dict con el evento;
    de lo contrario retorna None.

    El caller debe leer gains_a_aplicar despues de cada evento y actualizar
    el PID si los gains cambiaron.
    """

    def __init__(self, gains_init, steps, steps_min, step_decay,
                 eval_frames, mejora_min, mse_emergencia, gains_min, gains_max):
        self.gains         = list(gains_init)
        self.best_gains    = list(gains_init)
        self.best_mse      = float("inf")
        self.steps         = list(steps)
        self.steps_min     = list(steps_min)
        self.decay         = step_decay
        self.eval_frames   = eval_frames
        self.mejora_min    = mejora_min
        self.mse_emerg     = mse_emergencia
        self.gains_min     = list(gains_min)
        self.gains_max     = list(gains_max)

        self.param_idx     = 0
        self.fase          = "baseline"   # "baseline" | "plus" | "minus"
        self.frames_fase   = 0
        self.errs_fase: list[float] = []
        self.baseline_mse  = float("inf")
        self.gains_prueba  = list(gains_init)
        self.converged     = False
        self.epochs        = 0

    @property
    def gains_a_aplicar(self) -> list:
        return self.gains_prueba[:]

    def procesar(self, error: float, valido: bool = True) -> dict | None:
        """Ingerir un error de carril. valido=False descarta el frame (ej: decay sin deteccion)."""
        if not valido:
            return None
        self.errs_fase.append(error)
        self.frames_fase += 1

        # Seguridad: si MSE reciente es demasiado alto, revertir de inmediato
        if self.fase != "baseline" and self.frames_fase >= _VENTANA_EMERGENCIA:
            recientes = np.array(self.errs_fase[-_VENTANA_EMERGENCIA:])
            if float(np.mean(recientes ** 2)) > self.mse_emerg:
                return self._revertir_emergencia()

        if self.frames_fase >= self.eval_frames:
            return self._evaluar()
        return None

    # ── Internos ────────────────────────────────────────────────────────────────

    def _mse(self) -> float:
        return float(np.mean(np.array(self.errs_fase) ** 2)) if self.errs_fase else 0.0

    def _reset_fase(self) -> None:
        self.frames_fase = 0
        self.errs_fase   = []

    def _revertir_emergencia(self) -> dict:
        mse_p = float(np.mean(np.array(self.errs_fase[-_VENTANA_EMERGENCIA:]) ** 2))
        self.gains_prueba = self.gains[:]
        self._reset_fase()
        self.fase = "baseline"
        return {"tipo": "emergencia", "mse_parcial": round(mse_p, 5),
                "param": _NOMBRES[self.param_idx]}

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

        # fase == "minus"
        if mse < self.baseline_mse * (1.0 - self.mejora_min):
            return self._aceptar(mse, "-")
        return self._rechazar(mse)

    def _intentar(self, direction: str) -> dict:
        p    = self.param_idx
        sign = +1.0 if direction == "plus" else -1.0
        val  = self.gains[p] + sign * self.steps[p]
        val  = float(np.clip(val, self.gains_min[p], self.gains_max[p]))

        if abs(val - self.gains[p]) < 1e-9:
            # Limite alcanzado, probar la otra direccion o rendirse
            if direction == "plus":
                return self._intentar("minus")
            return self._rechazar(self.baseline_mse)

        self.gains_prueba       = self.gains[:]
        self.gains_prueba[p]    = round(val, 5)
        self.fase               = direction
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
            "mse_base":    round(self.baseline_mse, 5),
            "mse_trial":   round(mse_trial, 5),
            "nuevo_step":  round(self.steps[p], 5),
            "gains": self.gains[:],
        }
        self._avanzar()
        return ev

    def _avanzar(self) -> None:
        self.param_idx = (self.param_idx + 1) % 3
        self.fase      = "baseline"
        if self.param_idx == 0:
            self.epochs += 1
        # Convergencia: todos los steps en el minimo
        if all(s <= sm + 1e-9 for s, sm in zip(self.steps, self.steps_min)):
            self.converged = True


# ── Impresion de eventos ───────────────────────────────────────────────────────

def _imprimir_evento(ev: dict, auto: Autosintonizador) -> None:
    t    = _ts()
    tipo = ev["tipo"]

    if tipo == "nuevo_test":
        g = ev["gains_prueba"]
        d = "+" if ev["direction"] == "plus" else "-"
        print(f"\n[{t}] >> PROBAR {ev['param']}{d}{ev['step']:.4g} "
              f"| Kp={g[0]:.3f} Ki={g[1]:.4f} Kd={g[2]:.3f} "
              f"| MSE_base={ev['baseline_mse']:.5f}")

    elif tipo == "aceptado":
        g = ev["gains"]
        mejora = (ev["mse_base"] - ev["mse_nuevo"]) / max(ev["mse_base"], 1e-9) * 100
        print(f"[{t}] OK {ev['param']}{ev['direction']} "
              f"| {ev['mse_base']:.5f} -> {ev['mse_nuevo']:.5f} (-{mejora:.1f}%) "
              f"| Kp={g[0]:.3f} Ki={g[1]:.4f} Kd={g[2]:.3f}")

    elif tipo == "rechazado":
        g = ev["gains"]
        print(f"[{t}] NO {ev['param']:2s} "
              f"| base={ev['mse_base']:.5f} trial={ev['mse_trial']:.5f} "
              f"| step->{ev['nuevo_step']:.4g} "
              f"| Kp={g[0]:.3f} Ki={g[1]:.4f} Kd={g[2]:.3f}")

    elif tipo == "emergencia":
        print(f"[{t}] [!] EMERGENCIA {ev['param']} "
              f"| MSE_parcial={ev['mse_parcial']:.5f} > {_MSE_EMERGENCIA} "
              f"| revirtiendo a gains seguros")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay",   type=int,   default=10,
                        help="Segundos de countdown (default 10)")
    parser.add_argument("--rt",      type=int,   default=0,
                        help="Gas constante 0-255 (ej: 80). Omitir si usas cruise control")
    parser.add_argument("--dry-run", action="store_true",
                        help="No conectar gamepad (solo correr el algoritmo)")
    parser.add_argument("--kp",     type=float, default=_INIT[0])
    parser.add_argument("--ki",     type=float, default=_INIT[1])
    parser.add_argument("--kd",     type=float, default=_INIT[2])
    parser.add_argument("--eval-frames", type=int, default=_EVAL_FRAMES,
                        help=f"Frames por ventana de evaluacion (default {_EVAL_FRAMES})")
    parser.add_argument("--max-min", type=int, default=30,
                        help="Tiempo maximo en minutos (default 30)")
    parser.add_argument("--imgsz",  type=int,   default=_IMGSZ)
    parser.add_argument("--device", default=_DEVICE)
    args = parser.parse_args()

    rt_fijo  = max(0, min(255, args.rt))
    t_limite = time.monotonic() + args.max_min * 60

    # ── Countdown ──────────────────────────────────────────────────────────────
    if args.delay > 0:
        print("\nPon el camion en movimiento en una carretera abierta (40-80 km/h).")
        print(f"Arrancando en {args.delay}s...")
        for i in range(args.delay, 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1)
        print("  Iniciando autosintonizador!\n")

    # ── Componentes ────────────────────────────────────────────────────────────
    print(f"[{_ts()}] Cargando YOLOP (puede tardar ~60s en arranque en frio)...")
    yolop = InferenciaYOLOP(imgsz=args.imgsz, device=args.device)
    yolop.cargar()

    print(f"[{_ts()}] Warmup CUDA...")
    _dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for _ in range(3):
        yolop.procesar_frame(_dummy)
    print(f"[{_ts()}] Warmup listo.\n")

    pure_pursuit = PurePursuitVisual()
    fuente       = FuentePantalla(monitor=0, escalar_a=(1920, 1080))
    fuente.iniciar()

    pid = PIDController(kp=args.kp, ki=args.ki, kd=args.kd, limite=1.0)

    gamepad = None
    if not args.dry_run:
        try:
            import vgamepad as vg
            gamepad = vg.VX360Gamepad()
            print(f"[{_ts()}] Gamepad virtual conectado. RT={rt_fijo}")
        except Exception as e:
            print(f"[{_ts()}] [!] Gamepad no disponible: {e}. Modo dry-run.")

    auto = Autosintonizador(
        gains_init     = (args.kp, args.ki, args.kd),
        steps          = list(_STEP),
        steps_min      = list(_STEP_MIN),
        step_decay     = _DECAY,
        eval_frames    = args.eval_frames,
        mejora_min     = _MEJORA_MIN,
        mse_emergencia = _MSE_EMERGENCIA,
        gains_min      = list(_MIN),
        gains_max      = list(_MAX),
    )

    gains_pid_actual = auto.gains_a_aplicar

    print(f"[{_ts()}] === Autosintonizador iniciado ===")
    print(f"         Gains iniciales : Kp={args.kp:.3f}  Ki={args.ki:.4f}  Kd={args.kd:.3f}")
    print(f"         Eval frames     : {args.eval_frames}")
    print(f"         Mejora minima   : {_MEJORA_MIN*100:.0f}%")
    print(f"         Tiempo maximo   : {args.max_min} min")
    print(f"         Parar por Ctrl+C para ver resultado parcial.\n")

    # ── Loop principal ─────────────────────────────────────────────────────────
    desv_ema      = 0.0
    da_mask_cache = None
    ll_mask_cache = None
    yolop_cnt     = 0
    t_prev        = time.monotonic()
    n_frames      = 0
    n_decay       = 0

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

            # YOLOP cada _YOLOP_CADA frames
            yolop_cnt += 1
            if yolop_cnt >= _YOLOP_CADA or da_mask_cache is None:
                yolop_cnt = 0
                _, da_mask_cache, ll_mask_cache = yolop.procesar_frame(cuadro.imagen)

            da_mask     = da_mask_cache.copy()
            fila_roi    = int(da_mask.shape[0] * _ROI_FRAC)
            da_mask[:fila_roi, :]   = 0
            ll_mask_roi = ll_mask_cache.copy()
            ll_mask_roi[:fila_roi, :] = 0

            ll_px = int(np.count_nonzero(ll_mask_roi))
            giro, carril_perdido = pure_pursuit.calcular_giro(da_mask, ll_mask_roi)

            desv_ema  = _ALPHA_EMA * giro + (1.0 - _ALPHA_EMA) * desv_ema
            error_pid = 0.0 if abs(desv_ema) < _ZONA_MUERTA else float(desv_ema)

            # Actualizar PID si los gains cambiaron
            nuevos = auto.gains_a_aplicar
            if nuevos != gains_pid_actual:
                pid._kp, pid._ki, pid._kd = nuevos
                pid.reset()
                gains_pid_actual = nuevos

            stick = float(np.clip(pid.calcular(0.0, error_pid, dt), -1.0, 1.0))

            if gamepad is not None:
                _aplicar_gamepad(gamepad, stick, rt=rt_fijo)

            # Pasar error al autosintonizador (ignorar frames de decay)
            if carril_perdido:
                n_decay += 1
            evento = auto.procesar(giro, valido=not carril_perdido)

            if evento is not None:
                _imprimir_evento(evento, auto)
                # Resetear integral al cambiar de gains (evita contaminacion)
                if evento["tipo"] in ("nuevo_test", "emergencia"):
                    pid.reset()

            n_frames += 1

            # Status periodico cada 30 frames
            if n_frames % 30 == 0:
                errs_rec = auto.errs_fase[-30:] if len(auto.errs_fase) >= 5 else auto.errs_fase
                mse_p = float(np.mean(np.array(errs_rec) ** 2)) if errs_rec else 0.0
                g = auto.gains_a_aplicar
                src = "decay" if carril_perdido else ("ll" if ll_px > 800 else "da")
                decay_pct = int(100 * n_decay / n_frames) if n_frames else 0
                print(f"[{_ts()}] {auto.fase:8s} {_NOMBRES[auto.param_idx]:2s} "
                      f"| validos {auto.frames_fase:3d}/{auto.eval_frames} decay={decay_pct:2d}% "
                      f"| MSE={mse_p:.4f} stick={stick:+.3f} {src:5s} ll={ll_px:5d} "
                      f"| Kp={g[0]:.3f} Ki={g[1]:.4f} Kd={g[2]:.3f}"
                      f"{'  [MEJOR MSE='+f'{auto.best_mse:.4f}]' if auto.best_mse < float('inf') else ''}")

            # Control de loop
            elapsed  = time.monotonic() - t_now
            restante = _DT_LOOP - elapsed
            if restante > 0:
                time.sleep(restante)

    except KeyboardInterrupt:
        print(f"\n[{_ts()}] Interrumpido por Ctrl+C.")
    finally:
        if gamepad is not None:
            gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            gamepad.right_trigger(value=0)
            gamepad.left_trigger(value=0)
            gamepad.update()
        fuente.cerrar()

    # ── Resumen final ──────────────────────────────────────────────────────────
    decay_pct = int(100 * n_decay / n_frames) if n_frames else 0
    print(f"\n[{_ts()}] === Autosintonizador terminado ===")
    print(f"         Frames procesados : {n_frames}  (decay ignorado: {n_decay} = {decay_pct}%)")
    print(f"         Epocas completadas: {auto.epochs}")
    print(f"         Convergido        : {auto.converged}")

    # Gains aceptados = estado actual del algoritmo (lo que realmente se aplico al final)
    # Mejor MSE = el MSE mas bajo visto, pero puede ser de un segmento facil de carretera
    cg = auto.gains  # accepted gains — usar estos
    if auto.best_mse < float("inf"):
        bg = auto.best_gains
        print(f"\n  Gains aceptados (usar estos):")
        print(f"    Kp={cg[0]:.3f}  Ki={cg[1]:.4f}  Kd={cg[2]:.3f}")
        if bg != cg:
            print(f"  (Mejor MSE={auto.best_mse:.5f} fue con Kp={bg[0]:.3f} Ki={bg[1]:.4f} Kd={bg[2]:.3f}, "
                  f"pero puede ser de un tramo mas facil)")
        print()
        print("  Pega esta linea en src/control/gamepad_pid.py:")
        print(f"    _CFG_VOLANTE_DEFAULT = ConfigPID("
              f"kp={cg[0]:.3f}, ki={cg[1]:.4f}, kd={cg[2]:.3f})")
    else:
        print("         No se completo ninguna evaluacion.")
        print("         Intenta aumentar --delay y asegurar que el camion este en movimiento.")

    if auto.converged:
        print("\n  El algoritmo convergio. Los gains son optimos dentro del espacio de busqueda.")
    else:
        print(f"\n  Puedes continuar con:")
        start = cg if auto.best_mse < float("inf") else (args.kp, args.ki, args.kd)
        print(f"    python scripts/autosintonizar_pid_volante.py"
              f" --kp {start[0]:.3f} --ki {start[1]:.4f} --kd {start[2]:.3f}"
              f" --rt {rt_fijo} --delay 10")


if __name__ == "__main__":
    main()
