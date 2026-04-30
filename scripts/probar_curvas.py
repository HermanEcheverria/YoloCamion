"""Herramienta de diagnóstico de curvas para el sistema de lane-keeping ETS2.

Corre el pipeline YOLOP + PurePursuit sobre un video y genera imágenes de debug
anotadas con:
  - Máscara de área manejable (verde semitransparente)
  - Líneas de carril detectadas (rojo semitransparente)
  - Polinomios ajustados a las líneas (azul/naranja)
  - Punto de look-ahead (cian)
  - Curvatura estimada y fuente de señal activa

Uso rápido:
  python scripts/probar_curvas.py --video datos/videos/ets2_volvo_fh16.f140.m4a
  python scripts/probar_curvas.py --video mi_video.mp4 --salida datos/evidencia/curvas
  python scripts/probar_curvas.py --video mi_video.mp4 --cada 1  # analizar cada frame
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.percepcion.yolop_inference import InferenciaYOLOP
from src.control.pure_pursuit import PurePursuitVisual


def _dibujar_debug(
    frame: np.ndarray,
    da_mask: np.ndarray,
    ll_mask: np.ndarray,
    pure_pursuit: PurePursuitVisual,
    giro: float,
    carril_perdido: bool,
    n_frame: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    dbg = frame.copy()

    # Capas de máscara semitransparentes
    overlay = np.zeros_like(dbg)
    overlay[da_mask > 0] = (0, 200, 0)    # verde = área manejable
    overlay[ll_mask > 0] = (0, 50, 255)   # rojo = líneas de carril
    dbg = cv2.addWeighted(dbg, 0.75, overlay, 0.25, 0)

    # Líneas ROI (60% hacia abajo)
    y_roi = int(h * 0.60)
    cv2.line(dbg, (0, y_roi), (w, y_roi), (255, 255, 0), 1)
    cv2.putText(dbg, "ROI", (5, y_roi - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Ajuste polinomial de líneas (para visualización)
    x_camion = w // 2
    y_inicio = int(h * 0.60)
    ys_all, xs_all = np.where(ll_mask[y_inicio:, :] > 0)
    if len(ys_all) > 0:
        ys_all = ys_all + y_inicio
        izq_mask = xs_all < x_camion
        der_mask = xs_all >= x_camion
        ys_izq, xs_izq = ys_all[izq_mask], xs_all[izq_mask]
        ys_der, xs_der = ys_all[der_mask], xs_all[der_mask]

        _MIN_POLY = 30
        for (ys_l, xs_l, color) in [(ys_izq, xs_izq, (255, 120, 0)), (ys_der, xs_der, (0, 120, 255))]:
            if len(xs_l) >= _MIN_POLY:
                try:
                    poly = np.polyfit(ys_l, xs_l, 2)
                    y_vals = np.linspace(y_inicio, h - 1, 40).astype(int)
                    x_vals = np.polyval(poly, y_vals).astype(int)
                    pts = np.array([
                        (int(x), int(y)) for x, y in zip(x_vals, y_vals)
                        if 0 <= x < w
                    ])
                    if len(pts) > 1:
                        cv2.polylines(dbg, [pts.reshape(-1, 1, 2)], False, color, 2)
                except (np.linalg.LinAlgError, ValueError):
                    pass

    # Look-ahead point
    punto = pure_pursuit.ultimo_punto_debug
    if punto:
        cv2.circle(dbg, punto, 12, (0, 255, 255), -1)
        cv2.circle(dbg, punto, 14, (0, 0, 0), 2)

    # Centro del camión
    cv2.line(dbg, (x_camion, h - 1), (x_camion, int(h * 0.70)), (200, 200, 200), 1)

    # Barra de desviación (parte superior derecha)
    barra_x, barra_y, barra_w, barra_h = w - 220, 20, 200, 18
    cv2.rectangle(dbg, (barra_x, barra_y), (barra_x + barra_w, barra_y + barra_h), (50, 50, 50), -1)
    centro_barra = barra_x + barra_w // 2
    fill = int(giro * barra_w // 2)
    color_barra = (0, 255, 0) if abs(giro) < 0.2 else (0, 165, 255) if abs(giro) < 0.5 else (0, 0, 255)
    cv2.rectangle(dbg, (centro_barra, barra_y), (centro_barra + fill, barra_y + barra_h), color_barra, -1)
    cv2.rectangle(dbg, (barra_x, barra_y), (barra_x + barra_w, barra_y + barra_h), (200, 200, 200), 1)

    # Panel de texto
    curv = pure_pursuit.ultima_curvatura
    estado = "PERDIDO" if carril_perdido else ("CURVA" if curv > 0.4 else "RECTA")
    fuente = "poly" if (punto and ll_mask.any()) else ("da_mask" if not carril_perdido else "memoria")

    lineas_texto = [
        f"frame={n_frame}",
        f"giro={giro:+.3f}",
        f"curv={curv:.2f}",
        f"estado={estado}",
        f"fuente={fuente}",
    ]
    for i, txt in enumerate(lineas_texto):
        color_txt = (0, 0, 220) if estado == "PERDIDO" else (220, 220, 220)
        cv2.putText(dbg, txt, (10, 30 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(dbg, txt, (10, 30 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_txt, 1)

    return dbg


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico de curvas — YOLOP + Pure Pursuit")
    parser.add_argument("--video", required=True, help="Ruta al video ETS2 (mp4, m4v, m4a...)")
    parser.add_argument("--salida", default="datos/evidencia/curvas",
                        help="Directorio donde guardar las imágenes de debug")
    parser.add_argument("--cada", type=int, default=5,
                        help="Guardar debug cada N frames (default=5)")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = todos")
    parser.add_argument("--device", default="cuda", help="cuda o cpu")
    parser.add_argument("--mostrar", action="store_true",
                        help="Mostrar video en ventana OpenCV en tiempo real (requiere display)")
    args = parser.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    print(f"[curvas] Video: {args.video}")
    print(f"[curvas] Guardando debug cada {args.cada} frames en {salida}/")
    print(f"[curvas] Device: {args.device}")

    yolop = InferenciaYOLOP(device=args.device)
    print("[curvas] Cargando YOLOP...")
    yolop.cargar()
    pure_pursuit = PurePursuitVisual()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[curvas] ERROR: no se puede abrir {args.video}")
        sys.exit(1)

    n_frame = 0
    curvaturas = []
    perdidos = 0
    poly_activos = 0

    print("[curvas] Procesando...")
    t_inicio = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.max_frames > 0 and n_frame >= args.max_frames:
            break

        # Inferencia YOLOP
        _, da_mask, ll_mask = yolop.procesar_frame(frame)

        # Aplicar ROI (misma que el piloto)
        h, w = frame.shape[:2]
        y_roi = int(h * 0.60)
        da_mask_roi = da_mask.copy()
        da_mask_roi[:y_roi, :] = 0
        ll_mask_roi = ll_mask.copy()
        ll_mask_roi[:y_roi, :] = 0

        # Pure Pursuit
        giro, carril_perdido = pure_pursuit.calcular_giro(da_mask_roi, ll_mask_roi)
        curv = pure_pursuit.ultima_curvatura
        curvaturas.append(curv)
        if carril_perdido:
            perdidos += 1
        if pure_pursuit.ultimo_punto_debug and ll_mask_roi.any():
            poly_activos += 1

        # Guardar frame de debug
        if n_frame % args.cada == 0:
            dbg = _dibujar_debug(frame, da_mask_roi, ll_mask_roi,
                                 pure_pursuit, giro, carril_perdido, n_frame)
            ruta_img = salida / f"curva_debug_{n_frame:06d}.jpg"
            cv2.imwrite(str(ruta_img), dbg)

        if args.mostrar:
            dbg = _dibujar_debug(frame, da_mask_roi, ll_mask_roi,
                                 pure_pursuit, giro, carril_perdido, n_frame)
            cv2.imshow("Curvas Debug", dbg)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if n_frame % 100 == 0:
            elapsed = time.perf_counter() - t_inicio
            fps = (n_frame + 1) / elapsed if elapsed > 0 else 0
            print(f"  frame={n_frame} curv={curv:.2f} perdido={carril_perdido} fps={fps:.1f}")

        n_frame += 1

    cap.release()
    if args.mostrar:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_inicio
    print("\n=== Resumen ===")
    print(f"  Frames procesados : {n_frame}")
    print(f"  Tiempo total      : {elapsed:.1f}s ({n_frame/elapsed:.1f} FPS)")
    print(f"  Curvatura media   : {np.mean(curvaturas):.3f}")
    print(f"  Curvatura máx     : {np.max(curvaturas):.3f}")
    print(f"  Frames carril perdido: {perdidos} ({100*perdidos/max(n_frame,1):.1f}%)")
    print(f"  Frames polinomio activo: {poly_activos} ({100*poly_activos/max(n_frame,1):.1f}%)")
    print(f"  Imágenes guardadas en: {salida}/")
    print("\nCriterios OK para ETS2:")
    print(f"  Carril perdido <10%: {'OK' if perdidos/max(n_frame,1) < 0.10 else 'MEJORAR'}")
    print(f"  Polinomio >50%     : {'OK' if poly_activos/max(n_frame,1) > 0.50 else 'REVISAR'}")


if __name__ == "__main__":
    main()
