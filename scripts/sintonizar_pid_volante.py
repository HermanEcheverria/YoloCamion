"""Sintonizador interactivo de PID de volante para ETS2.

Captura frames en vivo, detecta carriles con YOLOP + PurePursuit,
aplica el PID de volante (_pid_vol) y envía el stick al gamepad virtual.
Muestra gráficas del error de carril y el stick en tiempo real.

Uso:
    python scripts/sintonizar_pid_volante.py
    python scripts/sintonizar_pid_volante.py --rt 80          # gas fijo 80/255
    python scripts/sintonizar_pid_volante.py --dry-run        # sin gamepad

Controles (ventana matplotlib activa):
    r       : resetear integrador PID
    s       : guardar CSV de la sesion actual
    q       : salir e imprimir gains recomendados

Como leer las graficas:
    Error de carril oscilando rapidamente → Kp muy alto o Kd muy bajo.
    Error convergente pero lento          → Kp muy bajo.
    Stick suave pero que no corrige       → aumentar Kp.
    MSE bajo y oscilaciones pocas         → gains correctos.

Al salir imprime los gains que tuvieron menor MSE del error de carril.
"""
import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fuente.pantalla import FuentePantalla
from src.control.pid import PIDController
from src.percepcion.yolop_inference import InferenciaYOLOP
from src.control.pure_pursuit import PurePursuitVisual

# ── Defaults ───────────────────────────────────────────────────────────────────
_KP_INIT   = 0.62
_KI_INIT   = 0.01
_KD_INIT   = 0.01
_DT_LOOP   = 0.033        # ~30 Hz
_VENTANA_S = 45.0         # segundos visibles en la grafica
_ALPHA_EMA = 0.45         # suavizado EMA del error de carril (igual que el piloto)
_ZONA_MUERTA = 0.015      # igual que _ZONA_MUERTA_CARRIL_PP en ejecutar_piloto.py
_ROI_FRAC  = 0.60         # enmascarar el top 60% (espejos/cielo)
_YOLOP_CADA = 2           # correr YOLOP cada N frames
_IMGSZ     = 864
_DEVICE    = "cuda"
# Frames que los gains deben permanecer estables antes de actualizar el mejor MSE.
# Evita que momentos de suerte (camion centrado por azar) contaminen la recomendacion.
_FRAMES_ESTABLE_MIN = 150  # ~5s a 30fps


def _actualizar_pid(pid: PIDController, kp: float, ki: float, kd: float) -> None:
    pid._kp = kp
    pid._ki = ki
    pid._kd = kd


def _aplicar_gamepad(gamepad, stick_x: float, rt: int = 0) -> None:
    gamepad.left_joystick_float(x_value_float=float(stick_x), y_value_float=0.0)
    gamepad.right_trigger(value=rt)
    gamepad.left_trigger(value=0)
    gamepad.update()


def _guardar_csv(historial: list, ruta: Path) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "error_pp", "ema", "stick", "fuente", "ll_px", "kp", "ki", "kd"])
        w.writerows(historial)
    print(f"CSV guardado: {ruta}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="No conectar gamepad (solo leer carril y simular PID)")
    parser.add_argument("--delay", type=int, default=5,
                        help="Segundos de espera antes de arrancar (default 5)")
    parser.add_argument("--rt", type=int, default=0,
                        help="Gas constante RT 0-255 para mover el camion (default 0). "
                             "Ej: --rt 80 para velocidad baja. "
                             "Nota: usa cruise control del juego para control de vel mas fino.")
    parser.add_argument("--imgsz", type=int, default=_IMGSZ)
    parser.add_argument("--device", default=_DEVICE)
    args = parser.parse_args()

    rt_fijo = max(0, min(255, int(args.rt)))

    # ── Countdown ──────────────────────────────────────────────────────────────
    if args.delay > 0:
        print(f"\nCambia al juego ETS2. Arrancando en {args.delay}s...")
        for i in range(args.delay, 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1)
        print("  ¡GO!\n")

    # ── Componentes ────────────────────────────────────────────────────────────
    print("Cargando YOLOP (puede tardar ~60s en arranque en frio)...")
    yolop = InferenciaYOLOP(imgsz=args.imgsz, device=args.device)
    yolop.cargar()

    print("Warmup CUDA...")
    _dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for _ in range(3):
        yolop.procesar_frame(_dummy)
    print("Warmup listo.\n")

    pure_pursuit = PurePursuitVisual()
    fuente = FuentePantalla(monitor=0, escalar_a=(1920, 1080))
    fuente.iniciar()

    pid = PIDController(kp=_KP_INIT, ki=_KI_INIT, kd=_KD_INIT, limite=1.0)

    gamepad = None
    if not args.dry_run:
        try:
            import vgamepad as vg
            gamepad = vg.VX360Gamepad()
            print("Gamepad virtual conectado.")
            if rt_fijo > 0:
                print(f"Gas fijo: RT={rt_fijo}/255")
        except Exception as e:
            print(f"[!] No se pudo conectar gamepad: {e}. Modo dry-run.")

    # ── Estado mutable ─────────────────────────────────────────────────────────
    state = {
        "running": True,
        "kp": _KP_INIT, "ki": _KI_INIT, "kd": _KD_INIT,
        "guardar": False,
        # Tracker de evaluacion: solo se actualiza si los gains llevan
        # >= _FRAMES_ESTABLE_MIN frames sin cambio (evita capturar suerte momentanea).
        "frames_estables": 0,
        "errs_ventana": [],   # errores acumulados desde el ultimo cambio de gains
    }
    historial: list = []

    # Mejor sesion (menor MSE)
    mejor_mse: float = float("inf")
    mejor_gains: tuple = (_KP_INIT, _KI_INIT, _KD_INIT)

    N = int(_VENTANA_S / _DT_LOOP) + 10
    buf_t     = deque(maxlen=N)
    buf_err   = deque(maxlen=N)   # error crudo de pure_pursuit
    buf_ema   = deque(maxlen=N)   # EMA del error
    buf_stick = deque(maxlen=N)   # stick enviado al gamepad

    t_inicio = time.monotonic()

    # ── Figura matplotlib ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title("Sintonizador PID Volante — ETS2")
    plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.38, hspace=0.35)

    ax_err   = fig.add_subplot(2, 1, 1)
    ax_stick = fig.add_subplot(2, 1, 2)

    ax_err.set_ylabel("Error de carril")
    ax_err.set_title("Error de carril (PurePursuit)")
    ax_err.set_ylim(-1.1, 1.1)
    ax_err.axhline(0, color="k", lw=0.5)
    ax_err.axhline( _ZONA_MUERTA, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax_err.axhline(-_ZONA_MUERTA, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax_err.grid(True, alpha=0.3)
    line_err, = ax_err.plot([], [], color="orange", lw=1.2, label="error PP")
    line_ema, = ax_err.plot([], [], "g-", lw=2,   label="EMA")
    ax_err.legend(loc="upper left", fontsize=8)
    txt_info = ax_err.text(
        0.01, 0.97, "", transform=ax_err.transAxes,
        va="top", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )

    ax_stick.set_ylabel("Stick X")
    ax_stick.set_xlabel("tiempo (s)")
    ax_stick.set_title("Comando de stick (salida del PID)")
    ax_stick.set_ylim(-1.1, 1.1)
    ax_stick.axhline(0, color="k", lw=0.5)
    ax_stick.grid(True, alpha=0.3)
    line_stick, = ax_stick.plot([], [], "b-", lw=1.5, label="stick")
    ax_stick.legend(loc="upper left", fontsize=8)

    # ── Sliders ────────────────────────────────────────────────────────────────
    ax_kp = fig.add_axes([0.12, 0.28, 0.75, 0.03])
    ax_ki = fig.add_axes([0.12, 0.22, 0.75, 0.03])
    ax_kd = fig.add_axes([0.12, 0.16, 0.75, 0.03])

    sl_kp = mwidgets.Slider(ax_kp, "Kp", 0.0, 3.0, valinit=_KP_INIT, valstep=0.01,  color="steelblue")
    sl_ki = mwidgets.Slider(ax_ki, "Ki", 0.0, 0.3, valinit=_KI_INIT, valstep=0.005, color="steelblue")
    sl_kd = mwidgets.Slider(ax_kd, "Kd", 0.0, 1.0, valinit=_KD_INIT, valstep=0.01,  color="steelblue")

    def on_slider(_val):
        state["kp"] = sl_kp.val
        state["ki"] = sl_ki.val
        state["kd"] = sl_kd.val
        _actualizar_pid(pid, state["kp"], state["ki"], state["kd"])
        # Reiniciar contador y ventana de evaluacion al cambiar gains.
        # El integral se resetea para no contaminar la nueva evaluacion con historia vieja.
        pid.reset()
        state["frames_estables"] = 0
        state["errs_ventana"] = []

    sl_kp.on_changed(on_slider)
    sl_ki.on_changed(on_slider)
    sl_kd.on_changed(on_slider)

    # ── Botones ────────────────────────────────────────────────────────────────
    ax_btn_reset = fig.add_axes([0.12, 0.08, 0.12, 0.05])
    ax_btn_save  = fig.add_axes([0.26, 0.08, 0.12, 0.05])
    ax_btn_quit  = fig.add_axes([0.40, 0.08, 0.12, 0.05])

    btn_reset = mwidgets.Button(ax_btn_reset, "Reset\nintegrador", color="lightyellow")
    btn_save  = mwidgets.Button(ax_btn_save,  "Guardar\nCSV",      color="lightgreen")
    btn_quit  = mwidgets.Button(ax_btn_quit,  "Salir",             color="salmon")

    def on_reset(_evt): pid.reset(); print("Integrador reseteado.")
    def on_save(_evt):  state["guardar"] = True
    def on_quit(_evt):  state["running"] = False

    btn_reset.on_clicked(on_reset)
    btn_save.on_clicked(on_save)
    btn_quit.on_clicked(on_quit)

    def on_key(event):
        k = event.key
        if k == "r":
            pid.reset(); print("  Integrador reseteado.")
        elif k == "s":
            state["guardar"] = True
        elif k in ("q", "escape"):
            state["running"] = False

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.ion()
    plt.show(block=False)

    # ── Loop principal ─────────────────────────────────────────────────────────
    desv_ema: float = 0.0
    da_mask_cache = None
    ll_mask_cache = None
    yolop_contador = 0
    t_prev = time.monotonic()
    n_frames = 0

    try:
        while state["running"]:
            t_now = time.monotonic()
            dt    = max(0.010, t_now - t_prev)
            t_prev = t_now
            t_rel  = t_now - t_inicio

            cuadro = fuente.siguiente()
            if cuadro is None:
                time.sleep(0.01)
                continue

            # YOLOP cada _YOLOP_CADA frames (igual que el piloto)
            yolop_contador += 1
            if yolop_contador >= _YOLOP_CADA or da_mask_cache is None:
                yolop_contador = 0
                _, da_mask_cache, ll_mask_cache = yolop.procesar_frame(cuadro.imagen)

            da_mask   = da_mask_cache.copy()
            ll_mask   = ll_mask_cache
            fila_roi  = int(da_mask.shape[0] * _ROI_FRAC)
            da_mask[:fila_roi, :] = 0
            ll_mask_roi = ll_mask.copy()
            ll_mask_roi[:fila_roi, :] = 0

            ll_px = int(np.count_nonzero(ll_mask_roi))
            fuente_carril = "ll" if ll_px > 800 else "da"

            # Pure Pursuit
            giro, carril_perdido = pure_pursuit.calcular_giro(da_mask, ll_mask_roi)
            if carril_perdido:
                fuente_carril = "decay"

            # EMA
            desv_ema = _ALPHA_EMA * giro + (1.0 - _ALPHA_EMA) * desv_ema
            # Decaer EMA cuando practicamente parado (evita acumulacion de sesgo)
            # — no tenemos velocidad aqui, pero si ema se aleja de 0 sin corrección
            #   significa que el carril se perdio. El decay de PurePursuit ya lo maneja.

            # PID sobre el error crudo (positivo = girar izq → salida negativa → stick izq)
            error_pid = 0.0 if abs(desv_ema) < _ZONA_MUERTA else float(desv_ema)
            stick = float(np.clip(pid.calcular(0.0, error_pid, dt), -1.0, 1.0))

            if gamepad is not None:
                _aplicar_gamepad(gamepad, stick, rt=rt_fijo)

            # Historial
            fila = [
                round(t_rel, 3), round(giro, 4), round(desv_ema, 4),
                round(stick, 4), fuente_carril, ll_px,
                state["kp"], state["ki"], state["kd"],
            ]
            historial.append(fila)

            buf_t.append(t_rel)
            buf_err.append(giro)
            buf_ema.append(desv_ema)
            buf_stick.append(stick)

            # Acumular en la ventana de evaluacion actual
            state["frames_estables"] += 1
            state["errs_ventana"].append(giro)

            n_frames += 1

            # Actualizar grafica cada 3 iteraciones (~10 Hz visual)
            if n_frames % 3 == 0:
                ts = list(buf_t)
                t0 = ts[-1] - _VENTANA_S if ts else 0.0

                line_err.set_data(ts, list(buf_err))
                line_ema.set_data(ts, list(buf_ema))
                line_stick.set_data(ts, list(buf_stick))

                for ax in (ax_err, ax_stick):
                    ax.set_xlim(t0, t0 + _VENTANA_S)

                # MSE de la ventana de visualizacion (solo informativo)
                errs = list(buf_err)
                mse_vis = float(np.mean(np.array(errs) ** 2)) if errs else 0.0
                n_cambios = sum(
                    1 for i in range(1, len(errs)) if errs[i - 1] * errs[i] < 0
                )

                # Evaluar mejor MSE SOLO si los gains llevan suficiente tiempo estables.
                # mse_eval usa SOLO los errores acumulados desde el ultimo cambio de gains.
                fe = state["frames_estables"]
                if fe >= _FRAMES_ESTABLE_MIN:
                    errs_eval = np.array(state["errs_ventana"])
                    mse_eval = float(np.mean(errs_eval ** 2))
                    if mse_eval < mejor_mse:
                        mejor_mse = mse_eval
                        mejor_gains = (state["kp"], state["ki"], state["kd"])
                    lbl_eval = f"MSE_eval={mse_eval:.4f}"
                else:
                    lbl_eval = f"estabilizando {fe}/{_FRAMES_ESTABLE_MIN}..."

                txt_info.set_text(
                    f"err={giro:+.3f}  ema={desv_ema:+.3f}  stick={stick:+.3f}  "
                    f"src={fuente_carril}  ll={ll_px}\n"
                    f"MSE_vis={mse_vis:.4f}  {lbl_eval}  osc={n_cambios}\n"
                    f"Kp={state['kp']:.2f} Ki={state['ki']:.3f} Kd={state['kd']:.3f}  "
                    f"mejor={mejor_mse:.4f}"
                )

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            if state["guardar"]:
                state["guardar"] = False
                ruta_csv = Path("datos/evidencia") / f"pid_vol_{int(t_inicio)}.csv"
                ruta_csv.parent.mkdir(parents=True, exist_ok=True)
                _guardar_csv(historial, ruta_csv)

            # Control de loop
            elapsed  = time.monotonic() - t_now
            restante = _DT_LOOP - elapsed
            if restante > 0:
                time.sleep(restante)

    except KeyboardInterrupt:
        print("\nInterrumpido por Ctrl+C.")
    finally:
        if gamepad is not None:
            gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            gamepad.right_trigger(value=0)
            gamepad.left_trigger(value=0)
            gamepad.update()
        fuente.cerrar()
        plt.close("all")

    # ── Resumen final ──────────────────────────────────────────────────────────
    print("\n=== Sesion terminada ===")
    if len(historial) < 30:
        print("  Pocas muestras — no hay suficientes datos para recomendar gains.")
        return

    arr_err = np.array([r[1] for r in historial])
    mse_final = float(np.mean(arr_err ** 2))
    print(f"  Frames registrados      : {len(historial)}")
    print(f"  MSE error carril total  : {mse_final:.5f}")

    if mejor_mse == float("inf"):
        print()
        print("  [!] No se completó ninguna evaluación estable.")
        print(f"      Cada set de gains debe permanecer >={_FRAMES_ESTABLE_MIN} frames")
        print("      sin cambio para ser evaluado. Usa los gains finales como referencia.")
        print()
        print("  Gains usados al final de la sesion:")
        print(f"    Kp = {state['kp']:.3f}")
        print(f"    Ki = {state['ki']:.4f}")
        print(f"    Kd = {state['kd']:.4f}")
    else:
        print(f"  MSE minimo evaluado     : {mejor_mse:.5f}")
        print()
        print(f"  Gains con menor MSE (recomendados — {_FRAMES_ESTABLE_MIN}+ frames estables):")
        print(f"    Kp = {mejor_gains[0]:.3f}")
        print(f"    Ki = {mejor_gains[1]:.4f}")
        print(f"    Kd = {mejor_gains[2]:.4f}")
        print()
        print("  Gains usados al final de la sesion:")
        print(f"    Kp = {state['kp']:.3f}")
        print(f"    Ki = {state['ki']:.4f}")
        print(f"    Kd = {state['kd']:.4f}")
        print()
        print("  Para aplicar los mejores gains edita src/control/gamepad_pid.py:")
        print(f"    _CFG_VOLANTE_DEFAULT = ConfigPID(kp={mejor_gains[0]:.3f},"
              f" ki={mejor_gains[1]:.4f}, kd={mejor_gains[2]:.4f})")

    ruta_csv = Path("datos/evidencia") / f"pid_vol_{int(t_inicio)}.csv"
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    _guardar_csv(historial, ruta_csv)


if __name__ == "__main__":
    main()
