"""Sintonizador interactivo de PID de velocidad para ETS2.

Captura frames en vivo, lee la velocidad del HUD, corre el PID de velocidad
y aplica RT al gamepad virtual. Muestra graficas en tiempo real y permite
ajustar Kp/Ki/Kd con sliders mientras el camion avanza.

Uso:
    python scripts/sintonizar_pid_velocidad.py
    python scripts/sintonizar_pid_velocidad.py --setpoint 60
    python scripts/sintonizar_pid_velocidad.py --dry-run   # sin gamepad

Controles (ventana matplotlib activa):
    1-8     : setpoint 10/20/30/40/50/60/70/80 km/h
    0       : setpoint 0 km/h (detener / soltar gas)
    r       : resetear integrador PID
    s       : guardar CSV de la sesion actual
    q       : salir e imprimir gains recomendados

Al salir imprime los gains que tuvieron menor error cuadratico medio.
"""
import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")       # backend que soporta sliders + eventos de teclado
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.percepcion.velocidad_dashboard import EstimadorVelocidadDashboard
from src.fuente.pantalla import FuentePantalla
from src.control.pid import PIDController

# ── Defaults ───────────────────────────────────────────────────────────────────
_KP_INIT  = 0.65
_KI_INIT  = 0.05
_KD_INIT  = 0.04
_MAX_KMH  = 90.0
_DT_LOOP  = 0.08          # ~12 Hz (suficiente para control de velocidad)
_VENTANA_S = 45.0         # segundos visibles en la grafica
_MARGEN_ACCEL = 0.04      # igual que gamepad_pid: no acelerar si vel+margen >= target
_VEL_MIN_LT   = 0.03      # no frenar si vel < esto (evita reversa)
_FRENO_FACTOR = 0.40      # LT proporcional al exceso normalizado


def _actualizar_pid(pid: PIDController, kp: float, ki: float, kd: float) -> None:
    pid._kp = kp
    pid._ki = ki
    pid._kd = kd


def _calcular_lt(vel_norm: float, target_norm: float) -> int:
    """Frenada proporcional cuando la velocidad supera el objetivo."""
    exceso = vel_norm - target_norm
    if exceso <= 0.02 or vel_norm < _VEL_MIN_LT:
        return 0
    return int(min(1.0, exceso * _FRENO_FACTOR) * 255)


def _aplicar_gamepad(gamepad, rt: int, lt: int) -> None:
    gamepad.right_trigger(value=rt)
    gamepad.left_trigger(value=lt)
    gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gamepad.update()


def _guardar_csv(historial: list, ruta: Path) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "setpoint_kmh", "actual_kmh", "error_kmh", "rt", "lt", "kp", "ki", "kd"])
        w.writerows(historial)
    print(f"CSV guardado: {ruta}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setpoint", type=float, default=50.0,
                        help="Velocidad objetivo inicial en km/h (default 50)")
    parser.add_argument("--max-kmh", type=float, default=_MAX_KMH,
                        help="km/h = velocidad normalizada 1.0 (default 90)")
    parser.add_argument("--dry-run", action="store_true",
                        help="No conectar gamepad (solo leer velocidad y simular PID)")
    parser.add_argument("--delay", type=int, default=5,
                        help="Segundos de espera antes de arrancar (default 5)")
    args = parser.parse_args()

    max_kmh = float(args.max_kmh)
    setpoint_kmh = float(args.setpoint)

    # ── Countdown ──────────────────────────────────────────────────────────────
    if args.delay > 0:
        print(f"\nCambia al juego ETS2. Arrancando en {args.delay}s...")
        for i in range(args.delay, 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1)
        print("  ¡GO!\n")

    # ── Componentes ────────────────────────────────────────────────────────────
    estimador = EstimadorVelocidadDashboard(max_kmh_norm=max_kmh, retener_frames=8)
    fuente    = FuentePantalla(monitor=0, escalar_a=(1920, 1080))
    fuente.iniciar()

    pid = PIDController(kp=_KP_INIT, ki=_KI_INIT, kd=_KD_INIT, limite=1.0)

    gamepad = None
    if not args.dry_run:
        try:
            import vgamepad as vg
            gamepad = vg.VX360Gamepad()
            print("Gamepad virtual conectado.")
        except Exception as e:
            print(f"[!] No se pudo conectar gamepad: {e}. Corriendo en modo dry-run.")

    # ── Estado mutable ─────────────────────────────────────────────────────────
    state = {
        "running":      True,
        "setpoint_kmh": setpoint_kmh,
        "kp":           _KP_INIT,
        "ki":           _KI_INIT,
        "kd":           _KD_INIT,
        "guardar":      False,
    }
    historial: list = []          # filas CSV
    mejores: dict = {}            # para recomendar gains al salir

    # Buffers de graficas (rolling window)
    N = int(_VENTANA_S / _DT_LOOP) + 10
    buf_t       = deque(maxlen=N)
    buf_sp      = deque(maxlen=N)
    buf_vel     = deque(maxlen=N)
    buf_error   = deque(maxlen=N)
    buf_rt      = deque(maxlen=N)
    buf_lt      = deque(maxlen=N)

    t_inicio = time.monotonic()

    # ── Figura matplotlib ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title("Sintonizador PID Velocidad — ETS2")

    # Reservar espacio para sliders y botones abajo
    plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.38, hspace=0.35)

    ax_vel = fig.add_subplot(2, 1, 1)
    ax_rt  = fig.add_subplot(2, 1, 2)

    # Velocidad
    ax_vel.set_ylabel("km/h")
    ax_vel.set_xlabel("")
    ax_vel.set_title("Velocidad")
    ax_vel.set_ylim(-5, max_kmh + 10)
    ax_vel.grid(True, alpha=0.3)
    line_sp,  = ax_vel.plot([], [], "b--", lw=1.5, label="setpoint")
    line_vel, = ax_vel.plot([], [], "g-",  lw=2,   label="actual")
    ax_vel.legend(loc="upper left", fontsize=8)
    txt_info = ax_vel.text(0.01, 0.97, "", transform=ax_vel.transAxes,
                           va="top", fontsize=8, family="monospace",
                           bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # RT / LT
    ax_rt.set_ylabel("Comando (%)")
    ax_rt.set_xlabel("tiempo (s)")
    ax_rt.set_title("Gas (RT) y Freno (LT)")
    ax_rt.set_ylim(-5, 105)
    ax_rt.axhline(0, color="k", lw=0.5)
    ax_rt.grid(True, alpha=0.3)
    line_rt, = ax_rt.plot([], [], "r-",  lw=1.5, label="RT (gas)")
    line_lt, = ax_rt.plot([], [], "c-",  lw=1.5, label="LT (freno)")
    ax_rt.legend(loc="upper left", fontsize=8)

    # ── Sliders ────────────────────────────────────────────────────────────────
    ax_kp = fig.add_axes([0.12, 0.28, 0.75, 0.03])
    ax_ki = fig.add_axes([0.12, 0.22, 0.75, 0.03])
    ax_kd = fig.add_axes([0.12, 0.16, 0.75, 0.03])

    sl_kp = mwidgets.Slider(ax_kp, "Kp", 0.0, 3.0, valinit=_KP_INIT, valstep=0.01, color="steelblue")
    sl_ki = mwidgets.Slider(ax_ki, "Ki", 0.0, 0.5, valinit=_KI_INIT, valstep=0.005, color="steelblue")
    sl_kd = mwidgets.Slider(ax_kd, "Kd", 0.0, 0.3, valinit=_KD_INIT, valstep=0.005, color="steelblue")

    def on_slider(_val):
        state["kp"] = sl_kp.val
        state["ki"] = sl_ki.val
        state["kd"] = sl_kd.val
        _actualizar_pid(pid, state["kp"], state["ki"], state["kd"])

    sl_kp.on_changed(on_slider)
    sl_ki.on_changed(on_slider)
    sl_kd.on_changed(on_slider)

    # ── Botones ────────────────────────────────────────────────────────────────
    ax_btn_reset  = fig.add_axes([0.12, 0.08, 0.12, 0.05])
    ax_btn_save   = fig.add_axes([0.26, 0.08, 0.12, 0.05])
    ax_btn_quit   = fig.add_axes([0.40, 0.08, 0.12, 0.05])

    btn_reset = mwidgets.Button(ax_btn_reset, "Reset\nintegrador", color="lightyellow")
    btn_save  = mwidgets.Button(ax_btn_save,  "Guardar\nCSV",      color="lightgreen")
    btn_quit  = mwidgets.Button(ax_btn_quit,  "Salir",             color="salmon")

    # Etiqueta de setpoint
    ax_sp_lbl = fig.add_axes([0.58, 0.07, 0.38, 0.07])
    ax_sp_lbl.axis("off")
    txt_sp_lbl = ax_sp_lbl.text(
        0.0, 0.5,
        f"Setpoint: {state['setpoint_kmh']:.0f} km/h  |  teclas 0-8 para cambiar",
        va="center", fontsize=9, family="monospace",
    )

    def on_reset(_evt):
        pid.reset()
        print("Integrador reseteado.")

    def on_save(_evt):
        state["guardar"] = True

    def on_quit(_evt):
        state["running"] = False

    btn_reset.on_clicked(on_reset)
    btn_save.on_clicked(on_save)
    btn_quit.on_clicked(on_quit)

    # ── Teclado ────────────────────────────────────────────────────────────────
    SETPOINTS_KMH = {
        "0": 0, "1": 10, "2": 20, "3": 30, "4": 40,
        "5": 50, "6": 60, "7": 70, "8": 80,
    }

    def on_key(event):
        k = event.key
        if k in SETPOINTS_KMH:
            state["setpoint_kmh"] = float(SETPOINTS_KMH[k])
            txt_sp_lbl.set_text(
                f"Setpoint: {state['setpoint_kmh']:.0f} km/h  |  teclas 0-8 para cambiar"
            )
            print(f"  Setpoint → {state['setpoint_kmh']:.0f} km/h")
        elif k == "r":
            pid.reset()
            print("  Integrador reseteado.")
        elif k == "s":
            state["guardar"] = True
        elif k in ("q", "escape"):
            state["running"] = False

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.ion()
    plt.show(block=False)

    # ── Loop principal ─────────────────────────────────────────────────────────
    t_prev   = time.monotonic()
    n_frames = 0

    try:
        while state["running"]:
            t_now = time.monotonic()
            dt    = max(0.01, t_now - t_prev)
            t_prev = t_now
            t_rel  = t_now - t_inicio

            # Captura y velocidad
            cuadro = fuente.siguiente()
            if cuadro is None:
                time.sleep(0.01)
                continue

            lectura = estimador.estimar(cuadro.imagen)
            vel_kmh  = float(lectura.kmh) if lectura.kmh is not None else 0.0
            vel_norm = lectura.norm

            sp_kmh  = state["setpoint_kmh"]
            sp_norm = min(1.0, sp_kmh / max_kmh)

            # PID → RT
            rt_norm = pid.calcular(sp_norm, vel_norm, dt)
            error_kmh = sp_kmh - vel_kmh

            # Calcular RT y LT
            if rt_norm > _MARGEN_ACCEL:
                rt = int(min(1.0, rt_norm) * 255)
            else:
                rt = 0
            lt = _calcular_lt(vel_norm, sp_norm)

            # Aplicar al gamepad
            if gamepad is not None:
                _aplicar_gamepad(gamepad, rt, lt)

            # Acumular historial
            fila = [
                round(t_rel, 3), round(sp_kmh, 1), round(vel_kmh, 1),
                round(error_kmh, 2), rt, lt,
                state["kp"], state["ki"], state["kd"],
            ]
            historial.append(fila)

            # Buffers de grafica
            buf_t.append(t_rel)
            buf_sp.append(sp_kmh)
            buf_vel.append(vel_kmh)
            buf_error.append(error_kmh)
            buf_rt.append(rt / 2.55)    # 0-100 %
            buf_lt.append(lt / 2.55)

            n_frames += 1

            # Actualizar grafica cada 3 iteraciones (~4 Hz visual)
            if n_frames % 3 == 0:
                ts  = list(buf_t)
                t0  = ts[-1] - _VENTANA_S if len(ts) > 0 else 0

                line_sp.set_data(ts, list(buf_sp))
                line_vel.set_data(ts, list(buf_vel))
                line_rt.set_data(ts, list(buf_rt))
                line_lt.set_data(ts, list(buf_lt))

                for ax in (ax_vel, ax_rt):
                    ax.set_xlim(t0, t0 + _VENTANA_S)

                # Texto de estado
                sse = np.mean(np.array(list(buf_error)) ** 2) if buf_error else 0.0
                txt_info.set_text(
                    f"vel={vel_kmh:5.1f} km/h  err={error_kmh:+5.1f}  "
                    f"RT={rt:3d}  LT={lt:3d}  "
                    f"MSE={sse:.2f}  "
                    f"Kp={state['kp']:.2f} Ki={state['ki']:.3f} Kd={state['kd']:.3f}"
                )

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # Guardar CSV si se pidió
            if state["guardar"]:
                state["guardar"] = False
                ruta_csv = Path("datos/evidencia") / f"pid_vel_{int(t_inicio)}.csv"
                ruta_csv.parent.mkdir(parents=True, exist_ok=True)
                _guardar_csv(historial, ruta_csv)

            # Control de loop
            elapsed = time.monotonic() - t_now
            restante = _DT_LOOP - elapsed
            if restante > 0:
                time.sleep(restante)

    except KeyboardInterrupt:
        print("\nInterrumpido por Ctrl+C.")
    finally:
        if gamepad is not None:
            gamepad.right_trigger(value=0)
            gamepad.left_trigger(value=0)
            gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            gamepad.update()
        fuente.cerrar()
        plt.close("all")

    # ── Resumen final ──────────────────────────────────────────────────────────
    print("\n=== Sesion terminada ===")
    if len(historial) < 10:
        print("  Pocas muestras — no hay suficiente datos para recomendar gains.")
        return

    arr = np.array([[r[2], r[3]] for r in historial])  # vel_kmh, error_kmh
    mse = float(np.mean(arr[:, 1] ** 2))
    print(f"  Frames registrados : {len(historial)}")
    print(f"  Error cuadratico medio (MSE) : {mse:.3f} km/h²")
    print()
    print(f"  Gains usados al final:")
    print(f"    Kp = {state['kp']:.3f}")
    print(f"    Ki = {state['ki']:.4f}")
    print(f"    Kd = {state['kd']:.4f}")
    print()
    print("  Para aplicar estos gains edita src/control/gamepad_pid.py:")
    print(f"    _CFG_VELOCIDAD_DEFAULT = ConfigPID(kp={state['kp']:.3f},"
          f" ki={state['ki']:.4f}, kd={state['kd']:.4f})")

    # Guardar CSV al salir si hay datos
    ruta_csv = Path("datos/evidencia") / f"pid_vel_{int(t_inicio)}.csv"
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    _guardar_csv(historial, ruta_csv)


if __name__ == "__main__":
    main()
