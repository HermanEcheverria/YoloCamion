import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


@dataclass
class Cuadro:
    imagen: np.ndarray
    timestamp: float
    indice: int
    fps_instantaneo: float


class Clase(Enum):
    VEHICULO = "vehiculo"
    MOTOCICLETA = "motocicleta"
    PEATON = "peaton"
    SEMAFORO = "semaforo"
    SENAL_ALTO = "senal_alto"
    DESCONOCIDO = "desconocido"


@dataclass
class Deteccion:
    clase: Clase
    caja: tuple[int, int, int, int]
    confianza: float
    area: int


@dataclass
class FisicaVisual:
    """Estimación visual de proximidad y velocidad relativa de un objeto rastreado.

    Pure-vision: TTC se deriva del crecimiento del bounding box y/o flujo óptico,
    nunca de telemetría interna. ttc_segundos = +inf cuando el objeto se aleja
    o no se aproxima de forma medible.
    """
    velocidad_relativa_px_s: float = 0.0
    ttc_segundos: float = math.inf
    area_px: int = 0
    area_anterior_px: int = 0
    centroide: tuple[int, int] = (0, 0)
    vector_flujo: tuple[float, float] = (0.0, 0.0)


@dataclass
class Seguimiento(Deteccion):
    id_seguimiento: int
    edad: int
    fisica: Optional[FisicaVisual] = None


class EstadoSemaforo(Enum):
    ROJO = "rojo"
    AMARILLO = "amarillo"
    VERDE = "verde"
    DESCONOCIDO = "desconocido"


class Region(Enum):
    FRENTE_CERCANO = "frente_cercano"
    FRENTE_LEJANO = "frente_lejano"
    ESPEJO_IZQ = "espejo_izq"
    ESPEJO_DER = "espejo_der"
    LATERAL_IZQ = "lateral_izq"
    LATERAL_DER = "lateral_der"


@dataclass
class EstadoEscena:
    frente_cercano_ocupado: bool
    frente_lejano_ocupado: bool
    peaton_en_riesgo: bool
    semaforo_visible: Optional[EstadoSemaforo]
    senal_alto_cercana: bool
    espejo_izq_ocupado: bool
    espejo_der_ocupado: bool
    vehiculos_totales: int
    confianza_percepcion: float
    timestamp: float
    ttc_minimo_frente_s: float = math.inf
    vehiculo_critico_id: Optional[int] = None


class Accion(Enum):
    MANTENER = "mantener"
    ACELERAR = "acelerar"
    FRENAR_SUAVE = "frenar_suave"
    FRENAR_FUERTE = "frenar_fuerte"
    ALTO_TOTAL = "alto_total"
    GIRAR_IZQ = "girar_izq"
    GIRAR_DER = "girar_der"
    REBASAR_IZQ = "rebasar_izq"
    REBASAR_DER = "rebasar_der"
    ESPERAR = "esperar"


@dataclass
class ComandoControl:
    acelerador: float
    freno: float
    volante: float
    timestamp: float


@dataclass
class SetpointControl:
    """Objetivo continuo que el FSM emite hacia los PIDs de la Capa 3.

    A diferencia de ComandoControl (salida instantánea), SetpointControl es
    una intención que el PID convierte en valores analógicos suaves.
    """
    velocidad_objetivo_norm: float = 0.0   # 0-1; 1 = velocidad máxima permitida
    freno_objetivo: float = 0.0            # 0-1; >=0.9 dispara bypass de emergencia
    desviacion_volante: float = 0.0        # -1..+1 (signo: <0 izq, >0 der) — stick command
    error_carril: float | None = None      # error crudo de PurePursuit (positivo = girar izq)
                                           # None cuando FSM controla el volante directamente
