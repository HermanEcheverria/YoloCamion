# src/control/pure_pursuit.py
import numpy as np


class PurePursuitVisual:
    """
    Controlador Pure Pursuit visual con look-ahead dinámico y detección por ll_mask.

    Fuentes de señal (en orden de prioridad):
    - Nivel 1a: extrapolación de líneas de carril (polyfit sobre todos los píxeles
      de ll_mask en el 60% inferior del frame, evaluado en la fila de look-ahead).
      Funciona aunque YOLOP solo detecte segmentos parciales de cada línea.
    - Nivel 1b: muestreo puntual de filas (fallback de L1a, comportamiento original).
    - Nivel 2: centroide de da_mask. Si se conoce el look-ahead previo, restringe
      la búsqueda a ±_VENTANA_CARRIL_FRAC alrededor de él para no cambiar de carril.
    - Nivel 3: último error × _DECAY cuando todo lo anterior falla.
    """

    _DECAY              = 0.85  # decaimiento por frame cuando carril perdido
    _BIAS_FRAC          = 0.0   # sin sesgo: centroide natural del área manejable
    _BIAS_CAM_PX        = 80    # cámara en cabina izquierda; centro físico del camión ≈ 80px a la derecha
    _FILA_LEJOS         = 0.62  # look-ahead en recta, sobre carretera visible
    _FILA_CERCA         = 0.70  # look-ahead en curva, antes del tablero/capo
    _CURVATURA_SCALE    = 0.0   # deshabilitado: look-ahead fijo en _FILA_LEJOS
    _ESCALA_ERROR       = 0.50  # normalisation factor: 50% frame width → full stick; more responsive on curves
    _MIN_LL_PIXELES     = 8     # mínimo por lado en muestreo puntual (L1b)
    _MIN_LL_FIT_PIXELES = 40    # mínimo por lado en extrapolación (L1a)
    _MAX_LL_PIXELES     = 200   # máximo total en L1c (segmentos, no píxeles — cap efectivamente inalcanzable)
    _MAX_LL_FIT_PIXELES = 8000  # máximo total en L1a
    _MAX_DELTA          = 0.10  # max cambio de error por frame; limita oscilacion
    _ALPHA_ANCLA_CARRIL = 0.15  # velocidad de actualización del ancla de carril
    _VENTANA_CARRIL_FRAC = 0.20  # L2: ±20 % del ancho alrededor del ancla
    _INNER_LL_FRAC       = 0.30  # L1: ventana por lado (ampliado de 0.20 para capturar línea izq en curvas)
    _HALF_LANE_PX        = 294  # semi-ancho de carril estimado para fallback de una sola línea (calibrado ETS2 Volvo FH16)
    _MAX_GAP_LINEA_PX    = 8
    _MAX_FRAMES_PUNTO_PERDIDO = 15

    def __init__(self) -> None:
        self._ultimo_error: float = 0.0
        self._ultimo_punto: tuple[int, int] | None = None
        self._frames_punto_perdido: int = 0
        self._x_ancla_carril: float | None = None
        self._ultima_curvatura: float = 0.0

    # ── API pública ────────────────────────────────────────────────────────────

    @property
    def ultimo_punto_debug(self) -> tuple[int, int] | None:
        return self._ultimo_punto

    @property
    def ultima_curvatura_debug(self) -> float:
        return self._ultima_curvatura

    _FILA_MAX = 0.74
    _SWEEP_PX = 40

    def calcular_giro(
        self,
        mascara_camino: np.ndarray,
        ll_mask: np.ndarray | None = None,
    ) -> tuple[float, bool]:
        """
        Calcula el error de dirección.

        Returns:
            (error_norm, carril_perdido)
            error_norm ∈ [-1, 1]: positivo → girar izquierda, negativo → girar derecha
        """
        alto, ancho = mascara_camino.shape
        # x_camion: centro del frame (posición de la cámara).
        # x_ref: centro físico del camión; la cámara está en la cabina izquierda,
        # así que el centro del chasis queda ~_BIAS_CAM_PX píxeles a la derecha.
        x_camion = ancho // 2
        # En ETS2 la vista ya esta calibrada para conducir por el centro visual
        # del parabrisas. Aplicar offset fisico a ll_mask sesga el objetivo
        # hacia la izquierda y empuja contra la barrera.
        x_ref_L1 = x_camion
        x_ref_L2 = x_camion

        curvatura = self._estimar_curvatura(mascara_camino, alto, ancho, ll_mask)
        self._ultima_curvatura = curvatura
        fila_base = int(alto * (
            self._FILA_LEJOS + curvatura * (self._FILA_CERCA - self._FILA_LEJOS)
        ))

        offsets = [-20, -10, 0, 10, 20]
        pesos   = [0.10, 0.20, 0.40, 0.20, 0.10]
        fila_max = int(alto * self._FILA_MAX)

        # ── Nivel 1: ll_mask ──────────────────────────────────────────────────
        if ll_mask is not None:
            # x_split = centro físico del camión (con corrección de cámara).
            # Las líneas del carril actual deben estar a ambos lados del camión.
            x_split = x_ref_L1

            # L1a: busca par de marcas que encierre al camión.
            # Si no se encuentra en la fila de look-ahead nominal, reintenta
            # progresivamente más lejos (hacia el horizonte, valores de y menores).
            centro_ll = None
            fila_ll_usada = fila_base
            for retroceso in (0, -30, -60, -90):
                filas_try = [max(0, min(fila_base + retroceso + off, alto - 1))
                             for off in offsets]
                centro_ll = self._centro_desde_ll_pares(ll_mask, filas_try, x_split, ancho)
                if centro_ll is not None:
                    fila_ll_usada = fila_base + retroceso
                    break

            filas_ll = [max(0, min(fila_ll_usada + off, alto - 1)) for off in offsets]

            # L1b: extrapolación por polyfit cuando las filas puntuales no bastan.
            if centro_ll is None:
                centro_ll = self._centro_desde_ll_fit(ll_mask, fila_base, x_split, ancho)

            # L1c: muestreo puntual simple (fallback)
            if centro_ll is None:
                centro_ll = self._centro_desde_ll(ll_mask, filas_ll, x_split)

            centro_ll = self._validar_y_actualizar_ancla(centro_ll, ancho)
            if centro_ll is not None:
                self._frames_punto_perdido = 0
                self._ultimo_punto = (int(round(centro_ll)), fila_ll_usada)
                dx = x_ref_L1 - centro_ll
                error = float(np.clip(dx / (ancho * self._ESCALA_ERROR), -1.0, 1.0))
                error = self._ultimo_error + float(np.clip(
                    error - self._ultimo_error, -self._MAX_DELTA, self._MAX_DELTA
                ))
                self._ultimo_error = error
                return error, False

        # ── Nivel 2: centroide da_mask ────────────────────────────────────────
        # Buscar alrededor del ancla del carril; si no hay ancla, usar centro del frame.
        x_busqueda = (int(round(self._x_ancla_carril))
                      if self._x_ancla_carril is not None else x_ref_L2)

        x_sum, w_sum, fila_usada = 0.0, 0.0, fila_base
        fila_try = fila_base
        while fila_try <= fila_max:
            xs, ws = 0.0, 0.0
            for off, peso in zip(offsets, pesos):
                y = max(0, min(fila_try + off, alto - 1))
                x = self._centroide_ventana(mascara_camino, y, x_busqueda, ancho)
                if x is None:
                    x = self._centroide_con_bias(mascara_camino, y, ancho)
                if x is not None:
                    xs += x * peso
                    ws += peso
            if ws > 0.0:
                x_sum, w_sum, fila_usada = xs, ws, fila_try
                break
            fila_try += self._SWEEP_PX

        if w_sum == 0.0:
            self._frames_punto_perdido += 1
            if self._frames_punto_perdido > self._MAX_FRAMES_PUNTO_PERDIDO:
                self._ultimo_punto = None
            self._ultimo_error *= self._DECAY
            return self._ultimo_error, True

        x_obj = x_sum / w_sum
        self._frames_punto_perdido = 0
        if self._x_ancla_carril is None:
            self._x_ancla_carril = float(x_obj)
        self._ultimo_punto = (int(round(x_obj)), fila_usada)

        dx = x_ref_L2 - x_obj
        error = float(np.clip(dx / (ancho * self._ESCALA_ERROR), -1.0, 1.0))
        error = self._ultimo_error + float(np.clip(
            error - self._ultimo_error, -self._MAX_DELTA, self._MAX_DELTA
        ))
        self._ultimo_error = error
        return error, False

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _validar_y_actualizar_ancla(self, centro: float | None, ancho: int) -> float | None:
        if centro is None:
            return None
        if self._x_ancla_carril is None:
            self._x_ancla_carril = float(centro)
            return centro
        # Descartar detecciones que saltan más de un carril (50 % del ancho)
        if abs(float(centro) - self._x_ancla_carril) > ancho * 0.50:
            return None
        self._x_ancla_carril = (
            (1.0 - self._ALPHA_ANCLA_CARRIL) * self._x_ancla_carril
            + self._ALPHA_ANCLA_CARRIL * float(centro)
        )
        return centro

    def _segmentos_ll_fila(self, fila: np.ndarray) -> list[float]:
        indices = np.nonzero(fila)[0]
        if len(indices) == 0:
            return []

        cortes = np.where(np.diff(indices) > self._MAX_GAP_LINEA_PX)[0] + 1
        grupos = np.split(indices, cortes)
        return [float((grupo[0] + grupo[-1]) / 2.0) for grupo in grupos if len(grupo) > 0]

    def _centro_desde_ll_pares(
        self,
        ll_mask: np.ndarray,
        filas: list[int],
        x_ref: int,
        ancho_frame: int,
    ) -> float | None:
        centros: list[float] = []
        min_w = ancho_frame * 0.08
        max_w = ancho_frame * 0.50

        for fila_y in filas:
            if fila_y < 0 or fila_y >= ll_mask.shape[0]:
                continue
            segmentos = sorted(self._segmentos_ll_fila(ll_mask[fila_y, :]))
            if len(segmentos) < 2:
                continue

            half = ancho_frame * self._INNER_LL_FRAC
            candidatos: list[tuple[float, float]] = []
            for izq, der in zip(segmentos, segmentos[1:]):
                ancho_carril = der - izq
                # Par válido: encierra al camión Y ambas líneas dentro de ±_INNER_LL_FRAC
                if (min_w <= ancho_carril <= max_w
                        and izq < x_ref < der
                        and izq >= x_ref - half
                        and der <= x_ref + half):
                    centro = (izq + der) / 2.0
                    candidatos.append((centro, ancho_carril))

            if not candidatos:
                continue

            # Si hay varios pares válidos, usar el más ancho (más probable que sea el carril)
            centro, _ = max(candidatos, key=lambda c: c[1])
            centros.append(float(centro))

        if len(centros) < 3:
            return None
        return float(np.median(np.asarray(centros)))

    def _centro_desde_ll_fit(
        self,
        ll_mask: np.ndarray,
        fila_objetivo: int,
        x_split: int,
        ancho_frame: int,
    ) -> float | None:
        """
        Ajusta una recta (polyfit grado 1) a todos los píxeles de ll_mask en el
        60 % inferior del frame y la evalúa en fila_objetivo.

        Funciona aunque YOLOP solo detecte segmentos parciales de cada línea de
        carril: extrapola la tendencia observada hasta el punto de look-ahead.
        """
        alto = ll_mask.shape[0]
        y0 = int(alto * 0.40)
        inner_half = int(ancho_frame * self._INNER_LL_FRAC)
        xs_izq: list[float] = []
        ys_izq: list[int] = []
        xs_der: list[float] = []
        ys_der: list[int] = []

        for y in range(y0, alto):
            segmentos = self._segmentos_ll_fila(ll_mask[y, :])
            if not segmentos:
                continue

            candidatos_izq = [
                centro for centro in segmentos
                if x_split - inner_half <= centro < x_split
            ]
            candidatos_der = [
                centro for centro in segmentos
                if x_split <= centro <= x_split + inner_half
            ]
            if candidatos_izq:
                xs_izq.append(max(candidatos_izq))
                ys_izq.append(y)
            if candidatos_der:
                xs_der.append(min(candidatos_der))
                ys_der.append(y)

        if len(xs_izq) + len(xs_der) > self._MAX_LL_FIT_PIXELES:
            return None

        has_izq = len(xs_izq) >= self._MIN_LL_FIT_PIXELES
        has_der = len(xs_der) >= self._MIN_LL_FIT_PIXELES

        if not has_izq and not has_der:
            return None

        try:
            if has_izq and has_der:
                # Ambas líneas: centro exacto
                coef_izq = np.polyfit(np.asarray(ys_izq), np.asarray(xs_izq), 1)
                coef_der = np.polyfit(np.asarray(ys_der), np.asarray(xs_der), 1)
                x_izq = float(np.polyval(coef_izq, fila_objetivo))
                x_der = float(np.polyval(coef_der, fila_objetivo))
                if x_izq >= x_der:
                    return None
                lane_w = x_der - x_izq
                if not (ancho_frame * 0.08 <= lane_w <= ancho_frame * 0.50):
                    return None
                return (x_izq + x_der) / 2.0
            elif has_der:
                # Solo línea derecha: centro estimado con semi-ancho calibrado
                coef = np.polyfit(np.asarray(ys_der), np.asarray(xs_der), 1)
                x_der = float(np.polyval(coef, fila_objetivo))
                return x_der - self._HALF_LANE_PX
            else:
                # Solo línea izquierda: centro estimado con semi-ancho calibrado
                coef = np.polyfit(np.asarray(ys_izq), np.asarray(xs_izq), 1)
                x_izq = float(np.polyval(coef, fila_objetivo))
                return x_izq + self._HALF_LANE_PX
        except (np.linalg.LinAlgError, ValueError):
            return None

    def _centroide_ventana(
        self,
        mascara: np.ndarray,
        fila_y: int,
        x_centro: int,
        ancho: int,
    ) -> int | None:
        """
        Mediana de da_mask en fila_y restringida a ±_VENTANA_CARRIL_FRAC del frame
        alrededor de x_centro (último look-ahead conocido). Evita cambios de carril.
        """
        half = int(ancho * self._VENTANA_CARRIL_FRAC)
        x1 = max(0, x_centro - half)
        x2 = min(ancho, x_centro + half)
        ventana = mascara[fila_y, x1:x2]
        indices = np.nonzero(ventana)[0]
        if len(indices) == 0:
            return None
        return int(np.median(indices)) + x1

    def _centroide_con_bias(self, mascara: np.ndarray, fila_y: int, ancho: int) -> int | None:
        """Mediana de da_mask en fila_y (usado en L2 sin punto previo conocido)."""
        fila = mascara[fila_y, :]
        indices = np.nonzero(fila)[0]
        if len(indices) == 0:
            return None
        x_min = int(indices[0])
        x_max = int(indices[-1])
        ancho_area = x_max - x_min
        if ancho_area == 0:
            return x_min
        x_bias = x_min + int(ancho_area * self._BIAS_FRAC)
        indices_der = indices[indices >= x_bias]
        if len(indices_der) == 0:
            return int(np.median(indices))
        return int(np.median(indices_der))

    def _centroide_ll_fila(self, ll_mask: np.ndarray, fila_y: int) -> int | None:
        """Mediana de los píxeles de ll_mask en fila_y."""
        if fila_y < 0 or fila_y >= ll_mask.shape[0]:
            return None
        indices = np.nonzero(ll_mask[fila_y, :])[0]
        if len(indices) == 0:
            return None
        return int(np.median(indices))

    def _centroide_ll_fila_ventana(
        self, ll_mask: np.ndarray, fila_y: int, x_centro: int, half: int
    ) -> int | None:
        """Mediana de píxeles de ll_mask en fila_y restringida a [x_centro-half, x_centro+half]."""
        if fila_y < 0 or fila_y >= ll_mask.shape[0]:
            return None
        x0 = max(0, x_centro - half)
        x1 = min(ll_mask.shape[1], x_centro + half)
        indices = np.nonzero(ll_mask[fila_y, x0:x1])[0]
        if len(indices) == 0:
            return None
        return int(np.median(indices)) + x0

    def _estimar_curvatura(
        self,
        mascara: np.ndarray,
        alto: int,
        ancho: int,
        ll_mask: np.ndarray | None = None,
    ) -> float:
        """Curvatura ∈ [0, 1] basada en la diferencia horizontal entre fila 85 % y 72 %."""
        y_cerca = int(alto * 0.70)
        y_lejos = int(alto * 0.60)

        x_cerca: int | None = None
        x_lejos: int | None = None
        if ll_mask is not None:
            half = int(ancho * self._INNER_LL_FRAC)
            x_centro = ancho // 2
            x_cerca = self._centroide_ll_fila_ventana(ll_mask, y_cerca, x_centro, half)
            x_lejos = self._centroide_ll_fila_ventana(ll_mask, y_lejos, x_centro, half)

        if x_cerca is None:
            x_cerca = self._centroide_con_bias(mascara, y_cerca, ancho)
        if x_lejos is None:
            x_lejos = self._centroide_con_bias(mascara, y_lejos, ancho)

        if x_cerca is None or x_lejos is None:
            return 0.0

        return float(np.clip(abs(x_cerca - x_lejos) / ancho * self._CURVATURA_SCALE, 0.0, 1.0))

    def _centro_desde_ll(
        self,
        ll_mask: np.ndarray,
        filas: list[int],
        x_camion: int,
    ) -> float | None:
        """
        Centro geométrico del carril desde píxeles de ll_mask en filas específicas.
        Fallback de _centro_desde_ll_fit cuando el polyfit no dispone de suficientes puntos.
        """
        pixeles_izq: list[int] = []
        pixeles_der: list[int] = []
        alto = ll_mask.shape[0]
        ancho = ll_mask.shape[1]
        inner_half = int(ancho * self._INNER_LL_FRAC)
        for fila_y in filas:
            if fila_y < 0 or fila_y >= alto:
                continue
            for x in self._segmentos_ll_fila(ll_mask[fila_y, :]):
                if x_camion - inner_half <= x < x_camion:
                    pixeles_izq.append(int(round(x)))
                elif x_camion <= x <= x_camion + inner_half:
                    pixeles_der.append(int(round(x)))
        total_px = len(pixeles_izq) + len(pixeles_der)
        if total_px < self._MIN_LL_PIXELES * 2:
            return None
        if len(pixeles_izq) < self._MIN_LL_PIXELES or len(pixeles_der) < self._MIN_LL_PIXELES:
            return None
        borde_izq = int(np.max(pixeles_izq))
        borde_der = int(np.min(pixeles_der))
        ancho_frame = ll_mask.shape[1]
        lane_width = borde_der - borde_izq
        if not (int(ancho_frame * 0.10) <= lane_width <= int(ancho_frame * 0.44)):
            return None
        return (borde_izq + borde_der) / 2.0
