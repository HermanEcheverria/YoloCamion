from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

from src.tipos import Clase, Seguimiento

_COCO_A_CLASE: dict[int, Clase] = {
    0: Clase.PEATON,
    2: Clase.VEHICULO,
    3: Clase.MOTOCICLETA,
    5: Clase.VEHICULO,
    7: Clase.VEHICULO,
    9: Clase.SEMAFORO,
    11: Clase.SENAL_ALTO,
}


class Tracker:
    """Detecta y hace seguimiento de objetos usando ByteTrack integrado en Ultralytics."""

    def __init__(
        self,
        ruta_modelo: str | Path = "datos/modelos/yolo26n.pt",
        confianza_min: float = 0.35,
        imgsz: int = 640,
        device: str = "cuda",
    ):
        self._ruta = str(ruta_modelo)
        self._confianza_min = confianza_min
        self._imgsz = imgsz
        self._device = device
        self._modelo: Optional[YOLO] = None
        self._edades: dict[int, int] = defaultdict(int)

    def cargar(self) -> None:
        self._modelo = YOLO(self._ruta)

    def rastrear(self, imagen: np.ndarray) -> list[Seguimiento]:
        if self._modelo is None:
            raise RuntimeError("Llama a cargar() antes de rastrear()")

        resultados = self._modelo.track(
            imagen,
            conf=self._confianza_min,
            device=self._device,
            imgsz=self._imgsz,
            persist=True,
            verbose=False,
        )

        seguimientos: list[Seguimiento] = []
        ids_actuales: set[int] = set()

        cajas = resultados[0].boxes
        if cajas is None or len(cajas) == 0:
            return []

        tiene_ids = cajas.id is not None

        for i, caja in enumerate(cajas):
            id_coco = int(caja.cls[0])
            clase = _COCO_A_CLASE.get(id_coco, Clase.DESCONOCIDO)
            x1, y1, x2, y2 = (int(v) for v in caja.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            # Sin IDs de tracking (primer frame): ID temporal negativo para no contaminar edades
            track_id = int(cajas.id[i]) if tiene_ids else -(i + 1)

            if tiene_ids:
                self._edades[track_id] += 1
                ids_actuales.add(track_id)

            edad = self._edades[track_id] if tiene_ids else 1
            seguimientos.append(
                Seguimiento(
                    clase=clase,
                    caja=(x1, y1, x2, y2),
                    confianza=float(caja.conf[0]),
                    area=area,
                    id_seguimiento=track_id,
                    edad=edad,
                )
            )

        # Limpiar tracks que desaparecieron
        ids_viejos = set(self._edades.keys()) - ids_actuales
        for viejo in ids_viejos:
            del self._edades[viejo]

        return seguimientos
