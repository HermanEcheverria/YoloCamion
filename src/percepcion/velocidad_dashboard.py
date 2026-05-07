"""Lectura de velocidad desde el HUD de ETS2.

Evita usar flujo optico para decidir si el camion esta parado: en ETS2, mantener
LT en 0 km/h engrana reversa. El HUD tiene la velocidad numerica en una posicion
estable, asi que leemos ese valor con OpenCV sin dependencias OCR externas.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LecturaVelocidadDashboard:
    kmh: int | None
    norm: float
    confianza: float
    valido: bool


_ROI_DIGITOS = (0.0474, 0.9287, 0.0781, 0.9722)  # x1, y1, x2, y2 — solo digitos velocimetro ETS2 1920x1080
_SIZE_ROI = (80, 45)  # ancho, alto de referencia
_SIZE_DIGITO = (10, 14)


_DIGITOS_ASCII: dict[int, tuple[str, ...]] = {
    0: (
        "..######..",
        "..######..",
        ".########.",
        ".###...###",
        ".##....###",
        ".##....###",
        "###....###",
        "###....###",
        ".##....###",
        ".##....###",
        ".###...###",
        ".########.",
        "..#######.",
        "....###...",
    ),
    1: (
        ".....#####",
        ".....#####",
        "....######",
        "..########",
        "#####..###",
        "####...###",
        ".......###",
        ".......###",
        ".......###",
        ".......###",
        ".......###",
        ".......###",
        ".......###",
        ".......###",
    ),
    2: (
        "...######.",
        "...######.",
        "..########",
        "####...###",
        "####...###",
        ".......###",
        "......####",
        ".....####.",
        ".....####.",
        "....####..",
        "...####...",
        "..####....",
        "##########",
        "##########",
    ),
    3: (
        ".########.",
        ".########.",
        ".########.",
        "......###.",
        ".....###..",
        "....###...",
        "...#####..",
        "...######.",
        ".......###",
        ".......###",
        "###....###",
        ".########.",
        "..######..",
        "....##....",
    ),
    4: (
        "......###.",
        "......###.",
        ".....####.",
        "....#####.",
        "...######.",
        "...##.###.",
        "..##..###.",
        ".###..###.",
        ".###..###.",
        ".#########",
        "##########",
        ".#########",
        "......###.",
        "......###.",
    ),
    5: (
        ".########.",
        ".########.",
        ".########.",
        ".###......",
        ".###......",
        ".#######..",
        ".########.",
        ".###..####",
        ".......###",
        "###....###",
        "###....###",
        ".########.",
        ".#######..",
        "....##....",
    ),
    6: (
        "..#######.",
        "..#######.",
        ".########.",
        ".###......",
        ".##.......",
        ".######...",
        ".########.",
        ".###..####",
        ".......###",
        ".##....###",
        "###....###",
        ".########.",
        "..######..",
        "....##....",
    ),
    7: (
        "##########",
        "##########",
        "##########",
        ".......###",
        "......###.",
        "......###.",
        ".....###..",
        ".....###..",
        "....###...",
        "....###...",
        "...###....",
        "...###....",
        "..###.....",
        "..###.....",
    ),
    8: (
        "..#######.",
        "..#######.",
        ".########.",
        ".###...###",
        ".##....###",
        ".###..####",
        "..#######.",
        ".#########",
        ".##....###",
        "###.....##",
        "###....###",
        ".#########",
        "..#######.",
        "....###...",
    ),
    9: (
        "..######..",
        "..######..",
        ".########.",
        "###...###.",
        "###....###",
        "###....###",
        "###...####",
        ".########.",
        ".########.",
        "..######..",
        ".....###..",
        "....###...",
        "...###....",
        "..###.....",
    ),
}


def _plantilla(bits: tuple[str, ...]) -> np.ndarray:
    arr = np.array([[255 if c == "#" else 0 for c in row] for row in bits], dtype=np.uint8)
    return arr.reshape(-1).astype(np.float32) / 255.0


_PLANTILLAS = {digito: _plantilla(bits) for digito, bits in _DIGITOS_ASCII.items()}


class EstimadorVelocidadDashboard:
    """Lee km/h del HUD inferior izquierdo y lo normaliza a [0, 1]."""

    def __init__(self, max_kmh_norm: float = 90.0, retener_frames: int = 15) -> None:
        self._max_kmh_norm = max(1.0, float(max_kmh_norm))
        self._retener_frames = max(0, int(retener_frames))
        self._ultimo_kmh: int | None = None
        self._frames_sin_lectura = 0

    def estimar(self, frame_bgr: np.ndarray) -> LecturaVelocidadDashboard:
        lectura = self.leer(frame_bgr)
        if lectura.valido and lectura.kmh is not None:
            # Filtro de plausibilidad: rechaza caídas bruscas que parecen
            # lecturas parciales del OCR (p.ej. "48" leído como "4").
            # Acepta si: no hay lectura previa, nuevo >= 15, o nuevo >= 40% del anterior.
            plausible = (
                self._ultimo_kmh is None
                or lectura.kmh >= 15
                or lectura.kmh >= self._ultimo_kmh * 0.65
            )
            if plausible:
                self._ultimo_kmh = lectura.kmh
                self._frames_sin_lectura = 0
                return lectura

            # Lectura implausible (p.ej. "4" cuando íbamos a 44 km/h):
            # devolver retenedor SIN incrementar frames_sin_lectura para que
            # el retenedor no expire mientras el OCR sigue leyendo mal.
            if self._ultimo_kmh is not None:
                norm = min(1.0, max(0.0, self._ultimo_kmh / self._max_kmh_norm))
                return LecturaVelocidadDashboard(self._ultimo_kmh, norm, 0.0, False)

        # OCR falló (valido=False o kmh=None): sí consumir el retenedor
        self._frames_sin_lectura += 1
        if self._ultimo_kmh is not None and self._frames_sin_lectura <= self._retener_frames:
            norm = min(1.0, max(0.0, self._ultimo_kmh / self._max_kmh_norm))
            return LecturaVelocidadDashboard(self._ultimo_kmh, norm, 0.0, False)
        return LecturaVelocidadDashboard(None, 0.0, 0.0, False)

    def leer(self, frame_bgr: np.ndarray) -> LecturaVelocidadDashboard:
        roi = self._recortar_roi(frame_bgr)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

        # Dilatar para reconectar trazos fragmentados del font digital de ETS2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.dilate(mask, kernel, iterations=1)

        componentes = self._extraer_componentes(mask)
        if not componentes:
            return LecturaVelocidadDashboard(None, 0.0, 0.0, False)

        digitos: list[int] = []
        confs: list[float] = []
        for _x, comp in componentes:
            digito, conf = self._clasificar(comp)
            if conf < 0.25:
                continue
            digitos.append(digito)
            confs.append(conf)

        if not digitos:
            return LecturaVelocidadDashboard(None, 0.0, 0.0, False)

        kmh = int("".join(str(d) for d in digitos))
        if kmh > 140:
            return LecturaVelocidadDashboard(None, 0.0, 0.0, False)

        confianza = float(np.mean(confs))
        norm = min(1.0, max(0.0, kmh / self._max_kmh_norm))
        return LecturaVelocidadDashboard(kmh, norm, confianza, confianza >= 0.30)

    def _recortar_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x1f, y1f, x2f, y2f = _ROI_DIGITOS
        x1 = max(0, min(w - 1, int(round(w * x1f))))
        y1 = max(0, min(h - 1, int(round(h * y1f))))
        x2 = max(x1 + 1, min(w, int(round(w * x2f))))
        y2 = max(y1 + 1, min(h, int(round(h * y2f))))
        roi = frame_bgr[y1:y2, x1:x2]
        return cv2.resize(roi, _SIZE_ROI, interpolation=cv2.INTER_AREA)

    def _extraer_componentes(self, mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        componentes: list[tuple[int, np.ndarray]] = []
        for i in range(1, num):
            x, y, w, h, area = (int(v) for v in stats[i])
            if not (5 <= h <= 55) or not (3 <= w <= 45) or area < 12:
                continue
            comp = mask[y:y + h, x:x + w]
            comp = cv2.resize(comp, _SIZE_DIGITO, interpolation=cv2.INTER_NEAREST)
            componentes.append((x, comp))

        componentes.sort(key=lambda item: item[0])
        return componentes[:3]

    def _clasificar(self, comp: np.ndarray) -> tuple[int, float]:
        vec = (comp.reshape(-1).astype(np.float32) / 255.0)
        mejor_digito = 0
        mejor_score = -1.0
        segundo_score = -1.0
        for digito, plantilla in _PLANTILLAS.items():
            score = self._score(vec, plantilla)
            if score > mejor_score:
                segundo_score = mejor_score
                mejor_score = score
                mejor_digito = digito
            elif score > segundo_score:
                segundo_score = score
        # Rechazar si la diferencia entre el mejor y segundo candidato es demasiado
        # pequeña — indica ambigüedad (p.ej. "2" vs "8", "5" vs "6").
        if mejor_score - segundo_score < 0.10:
            return mejor_digito, 0.0
        return mejor_digito, float(mejor_score)

    @staticmethod
    def _score(a: np.ndarray, b: np.ndarray) -> float:
        if float(a.std()) == 0.0 or float(b.std()) == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
