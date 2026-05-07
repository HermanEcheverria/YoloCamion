"""Calibrador de _HALF_LANE_PX para PurePursuit.

Captura frames en vivo, corre YOLOP y mide el semi-ancho del carril ego
cuando ambas lineas son visibles. Ejecutar con el juego visible en pantalla
y el camion centrado en el carril.

Uso:
    python scripts/calibrar_ancho_carril.py
    python scripts/calibrar_ancho_carril.py --frames 60 --guardar-imgs

Al terminar imprime el valor recomendado para _HALF_LANE_PX.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.percepcion.yolop_inference import InferenciaYOLOP
from src.fuente.pantalla import FuentePantalla


_INNER_LL_FRAC = 0.30   # igual que PurePursuit._INNER_LL_FRAC
_MIN_LL_FIT    = 40      # igual que PurePursuit._MIN_LL_FIT_PIXELES
_MAX_GAP_PX    = 8       # igual que PurePursuit._MAX_GAP_LINEA_PX
_FILA_LOOK     = 0.66    # fila de evaluacion (look-ahead en recta)
_ROI_TOP       = 0.60    # mascara de espejos (igual que el piloto)
_ROI_FIT_TOP   = 0.40    # zona inferior para polyfit (igual que L1b)


def _segmentos(fila: np.ndarray) -> list[float]:
    idx = np.nonzero(fila)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > _MAX_GAP_PX)[0] + 1
    grupos = np.split(idx, cortes)
    return [float((g[0] + g[-1]) / 2.0) for g in grupos if len(g) > 0]


def medir_semiancho(ll_mask: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """
    Ajusta lineas por polyfit (igual que L1b de PurePursuit) y devuelve:
        (x_izq, x_der, semiancho)  en pixeles, o (None, None, None) si falla.
    """
    alto, ancho = ll_mask.shape
    x_split = ancho // 2
    inner_half = int(ancho * _INNER_LL_FRAC)
    fila_obj = int(alto * _FILA_LOOK)
    y0 = int(alto * _ROI_FIT_TOP)

    xs_izq, ys_izq = [], []
    xs_der, ys_der = [], []

    for y in range(y0, alto):
        segs = _segmentos(ll_mask[y, :])
        candidatos_izq = [x for x in segs if x_split - inner_half <= x < x_split]
        candidatos_der = [x for x in segs if x_split <= x <= x_split + inner_half]
        if candidatos_izq:
            xs_izq.append(max(candidatos_izq))
            ys_izq.append(y)
        if candidatos_der:
            xs_der.append(min(candidatos_der))
            ys_der.append(y)

    if len(xs_izq) < _MIN_LL_FIT or len(xs_der) < _MIN_LL_FIT:
        return None, None, None

    try:
        coef_izq = np.polyfit(np.asarray(ys_izq), np.asarray(xs_izq), 1)
        coef_der = np.polyfit(np.asarray(ys_der), np.asarray(xs_der), 1)
        x_izq = float(np.polyval(coef_izq, fila_obj))
        x_der = float(np.polyval(coef_der, fila_obj))
    except (np.linalg.LinAlgError, ValueError):
        return None, None, None

    if x_izq >= x_der:
        return None, None, None

    lane_w = x_der - x_izq
    if not (ancho * 0.08 <= lane_w <= ancho * 0.50):
        return None, None, None

    return x_izq, x_der, lane_w / 2.0


def anotar_frame(frame: np.ndarray, ll_mask: np.ndarray,
                 x_izq: float | None, x_der: float | None,
                 semiancho: float | None, n_ok: int, n_total: int) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()

    # Superponer mascara de lineas (azul)
    mc = np.zeros_like(out)
    mc[ll_mask > 0] = (255, 80, 0)
    out = cv2.addWeighted(out, 0.75, mc, 0.25, 0)

    # Linea horizontal de look-ahead
    y_la = int(h * _FILA_LOOK)
    cv2.line(out, (0, y_la), (w, y_la), (0, 255, 255), 1)

    if x_izq is not None and x_der is not None:
        cx = int((x_izq + x_der) / 2)
        cv2.line(out, (int(x_izq), y_la - 20), (int(x_izq), y_la + 20), (0, 255, 0), 3)
        cv2.line(out, (int(x_der), y_la - 20), (int(x_der), y_la + 20), (0, 0, 255), 3)
        cv2.circle(out, (cx, y_la), 8, (0, 255, 255), -1)
        cv2.putText(out, f"semiancho={semiancho:.0f}px  ancho={x_der - x_izq:.0f}px",
                    (int(x_izq) + 5, y_la - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    estado = f"frames validos: {n_ok}/{n_total}"
    cv2.putText(out, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, "verde=izq  rojo=der  cyan=centro  [Q]=salir",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=80,
                        help="Frames a capturar antes de mostrar resultado (default 80)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guardar-imgs", action="store_true",
                        help="Guarda una imagen anotada cada 10 frames validos en datos/evidencia/")
    args = parser.parse_args()

    print("\n=== Calibrador de ancho de carril ===")
    print(f"Se capturarán hasta {args.frames} frames.")
    print("Mantén el camión centrado en el carril durante la captura.")
    print("Presiona Q en la ventana de preview para terminar antes.\n")

    yolop = InferenciaYOLOP(imgsz=864, device=args.device)
    print("Cargando YOLOP...")
    yolop.cargar()

    fuente = FuentePantalla(monitor=0, escalar_a=(1920, 1080))
    fuente.iniciar()

    semianchos: list[float] = []
    n_frame = 0
    n_guardadas = 0
    ruta_salida = Path("datos/evidencia")
    ruta_salida.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow("Calibracion carril", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibracion carril", 960, 540)

    try:
        while n_frame < args.frames:
            cuadro = fuente.siguiente()
            if cuadro is None:
                time.sleep(0.02)
                continue

            _, _, ll_mask = yolop.procesar_frame(cuadro.imagen)

            # Misma mascara de ROI que el piloto
            fila_roi = int(ll_mask.shape[0] * _ROI_TOP)
            ll_mask[:fila_roi, :] = 0

            x_izq, x_der, semiancho = medir_semiancho(ll_mask)

            if semiancho is not None:
                semianchos.append(semiancho)
                print(f"  Frame {n_frame:3d}: x_izq={x_izq:6.1f}  x_der={x_der:6.1f}"
                      f"  semiancho={semiancho:5.1f}px  (n={len(semianchos)})")

                if args.guardar_imgs and len(semianchos) % 10 == 0:
                    anotado = anotar_frame(cuadro.imagen, ll_mask, x_izq, x_der, semiancho,
                                           len(semianchos), n_frame + 1)
                    ruta_img = ruta_salida / f"calib_carril_{n_guardadas:03d}.jpg"
                    cv2.imwrite(str(ruta_img), anotado)
                    n_guardadas += 1
                    print(f"    → imagen guardada: {ruta_img}")

            anotado = anotar_frame(cuadro.imagen, ll_mask, x_izq, x_der, semiancho,
                                    len(semianchos), n_frame + 1)
            cv2.imshow("Calibracion carril", anotado)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Salida por usuario.")
                break

            n_frame += 1

    finally:
        fuente.cerrar()
        cv2.destroyAllWindows()

    # ── Resultado ────────────────────────────────────────────────────────────
    print("\n=== Resultado ===")
    if len(semianchos) < 2:
        print(f"Solo se obtuvieron {len(semianchos)} mediciones validas (minimo 5).")
        print("Posibles causas:")
        print("  - YOLOP no detecta ambas lineas al mismo tiempo")
        print("  - El camion no estaba centrado en el carril")
        print("  - La ruta tiene pocos marcadores visibles")
        print("\nPrueba en una autopista con marcas claras y camion centrado.")
        return

    arr = np.array(semianchos)
    p25, mediana, p75 = np.percentile(arr, [25, 50, 75])
    media = arr.mean()
    std = arr.std()

    print(f"  Mediciones validas : {len(arr)} / {n_frame}")
    print(f"  Media              : {media:.1f} px")
    print(f"  Mediana            : {mediana:.1f} px")
    print(f"  P25-P75            : {p25:.1f} – {p75:.1f} px")
    print(f"  Desv. estandar     : {std:.1f} px")
    print()
    recomendado = int(round(mediana))
    print(f"  Valor actual  _HALF_LANE_PX = 200")
    print(f"  Recomendado   _HALF_LANE_PX = {recomendado}")
    print()
    if abs(recomendado - 200) < 20:
        print("  El valor actual (200) ya es correcto — diferencia < 20 px.")
    else:
        print(f"  → Actualizar src/control/pure_pursuit.py línea con _HALF_LANE_PX")
        print(f"    Cambiar 200 por {recomendado}.")


if __name__ == "__main__":
    main()
