"""Calibracion interactiva del ROI del velocimetro ETS2.

Captura un frame del juego, muestra una ventana donde puedes arrastrar
un rectangulo sobre los digitos del velocimetro, y guarda las coordenadas
en src/percepcion/velocidad_dashboard.py.

Uso:
    python scripts/calibrar_roi_velocidad.py

Controles en la ventana:
    Arrastra con el mouse  — seleccionar ROI
    ENTER / ESPACIO        — confirmar seleccion
    C                      — cancelar y volver a redibujar
    ESC                    — salir sin guardar
"""
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fuente.pantalla import FuentePantalla
from src.percepcion.velocidad_dashboard import (
    _PLANTILLAS,
    _ROI_DIGITOS,
    _SIZE_DIGITO,
    _SIZE_ROI,
)

# ── Capturar frame ─────────────────────────────────────────────────────────────
print("Cambia a ETS2. Capturando en 3s...")
time.sleep(3)

fuente = FuentePantalla(monitor=0, escalar_a=(1920, 1080))
fuente.iniciar()

cuadro = None
for _ in range(10):
    cuadro = fuente.siguiente()
    if cuadro is not None:
        break
    time.sleep(0.05)
fuente.cerrar()

if cuadro is None:
    print("[!] No se pudo capturar frame.")
    sys.exit(1)

frame = cuadro.imagen
h, w = frame.shape[:2]
print(f"Frame: {w}x{h}")

# ROI actual marcada en azul para referencia
frame_ref = frame.copy()
ax1 = int(w * _ROI_DIGITOS[0]); ay1 = int(h * _ROI_DIGITOS[1])
ax2 = int(w * _ROI_DIGITOS[2]); ay2 = int(h * _ROI_DIGITOS[3])
cv2.rectangle(frame_ref, (ax1, ay1), (ax2, ay2), (255, 100, 0), 2)
cv2.putText(frame_ref, "ROI actual (azul)", (ax1, ay1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

# ── Escalar para que quepa en pantalla ────────────────────────────────────────
escala = min(1.0, 1280 / w, 720 / h)
dw, dh = int(w * escala), int(h * escala)
display = cv2.resize(frame_ref, (dw, dh))

# ── Seleccion interactiva ──────────────────────────────────────────────────────
print("\nArrastra el mouse sobre los DIGITOS del velocimetro.")
print("ENTER / ESPACIO = confirmar  |  C = reintentar  |  ESC = salir sin guardar\n")

roi_sel = cv2.selectROI(
    "Selecciona el velocimetro (ENTER=ok  C=reintentar  ESC=salir)",
    display,
    fromCenter=False,
    showCrosshair=True,
)
cv2.destroyAllWindows()

rx, ry, rw, rh = roi_sel
if rw == 0 or rh == 0:
    print("[!] Sin seleccion. Saliendo sin cambios.")
    sys.exit(0)

# ── Convertir a coordenadas reales y normalizadas ──────────────────────────────
x1_px = int(rx / escala)
y1_px = int(ry / escala)
x2_px = int((rx + rw) / escala)
y2_px = int((ry + rh) / escala)

x1f = round(x1_px / w, 4)
y1f = round(y1_px / h, 4)
x2f = round(x2_px / w, 4)
y2f = round(y2_px / h, 4)

print(f"ROI seleccionada:")
print(f"  Pixeles   : ({x1_px},{y1_px}) -> ({x2_px},{y2_px})")
print(f"  Fraccion  : ({x1f}, {y1f}, {x2f}, {y2f})")

# ── Analizar OCR con la nueva ROI ─────────────────────────────────────────────
roi_img = frame[y1_px:y2_px, x1_px:x2_px]
if roi_img.size == 0:
    print("[!] ROI vacia. Saliendo.")
    sys.exit(1)

roi_scaled = cv2.resize(roi_img, _SIZE_ROI, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(roi_scaled, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
mask_dil = cv2.dilate(mask, kernel, iterations=1)

num_cc, _lbl, stats_cc, _ = cv2.connectedComponentsWithStats(mask_dil, 8)
print(f"\nComponentes en mascara: {num_cc - 1} (sin fondo)")

digitos: list[int] = []
confs: list[float] = []
for i in range(1, num_cc):
    cx, cy, cw, ch, ca = (int(v) for v in stats_cc[i])
    pasa = (5 <= ch <= 55) and (3 <= cw <= 45) and ca >= 12
    comp = mask_dil[cy:cy + ch, cx:cx + cw]
    comp_r = cv2.resize(comp, _SIZE_DIGITO, interpolation=cv2.INTER_NEAREST)
    vec = comp_r.reshape(-1).astype(np.float32) / 255.0
    scores: dict[int, float] = {}
    for d, tpl in _PLANTILLAS.items():
        if float(vec.std()) == 0.0 or float(tpl.std()) == 0.0:
            scores[d] = 0.0
        else:
            scores[d] = float(np.corrcoef(vec, tpl)[0, 1])
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_d, best_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0.0
    diff = best_s - second_s
    if diff >= 0.10:
        estado = f"OK  -> digito={best_d}"
        if pasa:
            digitos.append(best_d)
            confs.append(best_s)
    else:
        estado = f"AMBIGUO (diff={diff:.2f} < 0.10)"
    top5 = "  ".join(f"{d}:{s:.2f}" for d, s in ranked[:5])
    print(f"  Comp {i}: x={cx} y={cy} w={cw} h={ch} area={ca}  pasa={pasa}")
    print(f"          Top5: {top5}")
    print(f"          → mejor={best_d} score={best_s:.3f} diff={diff:.3f}  {estado}")

if digitos:
    kmh = int("".join(str(d) for d in digitos))
    conf_media = float(np.mean(confs))
    print(f"\nLectura OCR con nueva ROI: kmh={kmh}  confianza={conf_media:.3f}")
else:
    print("\nLectura OCR: kmh=None  (digitos ambiguos o sin componentes validos)")

# ── Guardar imagenes debug ─────────────────────────────────────────────────────
Path("datos/debug").mkdir(parents=True, exist_ok=True)
marcado = frame.copy()
cv2.rectangle(marcado, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 3)
cv2.putText(marcado, f"Nueva ROI ({x1f},{y1f})-({x2f},{y2f})",
            (x1_px, max(0, y1_px - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imwrite("datos/debug/frame_roi_marcado.png", marcado)

roi_x4 = cv2.resize(roi_img, (roi_img.shape[1] * 4, roi_img.shape[0] * 4),
                    interpolation=cv2.INTER_NEAREST)
cv2.imwrite("datos/debug/roi_crop_x4.png", roi_x4)

mask_x4 = cv2.resize(mask_dil, (mask_dil.shape[1] * 4, mask_dil.shape[0] * 4),
                     interpolation=cv2.INTER_NEAREST)
cv2.imwrite("datos/debug/roi_mascara_x4.png", mask_x4)

print("\nImagenes guardadas en datos/debug/")

# ── Confirmar y guardar ────────────────────────────────────────────────────────
resp = input("\n¿Guardar esta ROI en velocidad_dashboard.py? [s/N]: ").strip().lower()
if resp != "s":
    print("Sin cambios.")
    sys.exit(0)

dashboard_path = Path("src/percepcion/velocidad_dashboard.py")
texto = dashboard_path.read_text(encoding="utf-8")
nueva_linea = (
    f"_ROI_DIGITOS = ({x1f}, {y1f}, {x2f}, {y2f})"
    f"  # x1, y1, x2, y2 — solo digitos velocimetro ETS2 1920x1080"
)
texto_nuevo = re.sub(
    r"_ROI_DIGITOS\s*=\s*\([^)]+\).*",
    nueva_linea,
    texto,
)
if texto_nuevo == texto:
    print("[!] No se encontro la linea _ROI_DIGITOS en el archivo. Sin cambios.")
    sys.exit(1)

dashboard_path.write_text(texto_nuevo, encoding="utf-8")
print(f"[OK] Guardado: _ROI_DIGITOS = ({x1f}, {y1f}, {x2f}, {y2f})")
print("\nVuelve a correr el script para verificar la lectura con la nueva ROI.")
