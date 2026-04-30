# src/control/pure_pursuit.py
import numpy as np


class PurePursuitVisual:
    """
    Controlador Pure Pursuit visual con look-ahead dinámico y detección por ll_mask.

    Fuentes de señal (en orden de prioridad):
    - Nivel 1a: Ajuste polinomial de ll_mask. Ajusta una parábola a TODOS los
      píxeles de línea visibles y predice el centro del carril a la distancia
      de anticipación. Es la señal más precisa en curvas porque usa la forma
      completa de la línea, no solo las filas del look-ahead.
    - Nivel 1b: ll_mask por filas (fallback cuando hay pocos píxeles totales).
      Calcula el centro exacto del carril actual como (borde_izq + borde_der) / 2.
    - Nivel 2: centroide de da_mask con barrido adaptativo (fallback).
    - Nivel 3: último error × 0.85 (memoria con decaimiento) cuando ambas fallan.

    Look-ahead dinámico: la fila de anticipación se acorta en curvas y se alarga
    en rectas, medido por la diferencia de centroides entre fila 72% y 85%.
    Suavizado multi-fila: promedia 5 filas con pesos gaussianos.
    """

    _DECAY = 0.85          # factor de decaimiento por frame cuando carril perdido
    _BIAS_FRAC = 0.30      # sesgo derecho: en vías bidireccionales da_mask cubre ambos
                           # carriles y el centroide cae en la línea central; con 0.30
                           # usamos la mediana del 70% derecho del área verde, que
                           # corresponde al carril derecho (conducción europea)
    _FILA_LEJOS = 0.72     # recta: zona media donde carretera lejana está clara sin espejos
    _FILA_CERCA = 0.85     # curva: más cerca pero no al ras del asfalto
    _CURVATURA_SCALE = 6.0 # factor de amplificación de la curvatura cruda
    _ESCALA_ERROR = 0.40   # normalización intermedia (look-ahead a distancia media)
    _MIN_LL_PIXELES = 8    # píxeles mínimos por lado en ll_mask para activar nivel 1b
    _MIN_LL_POLY = 30      # píxeles mínimos por lado para ajuste polinomial (nivel 1a)
    _MAX_DELTA = 0.28      # máximo cambio de error por llamada — suprime saltos de detección YOLOP
    _MIN_SIMETRIA = 0.08   # fracción mínima del lado minoritario de da_mask; por debajo se
                           # considera obstrucción estructural (pilar, paso inferior, valla) y
                           # se usa memoria con decaimiento en lugar de un centroide falso
    _MIN_PX_SIMETRIA = 300 # píxeles totales mínimos en da_mask para evaluar asimetría
    _ROI_POLY_FRAC = 0.60  # fracción superior de la imagen excluida del ajuste polinomial
                           # (igual que la ROI del piloto: espejos + capó)

    def __init__(self) -> None:
        self._ultimo_error: float = 0.0
        self._ultimo_punto: tuple[int, int] | None = None
        self._ultima_curvatura: float = 0.0

    # ── API pública ────────────────────────────────────────────────────────────

    @property
    def ultimo_punto_debug(self) -> tuple[int, int] | None:
        """Último look-ahead point calculado; None si el carril estaba perdido."""
        return self._ultimo_punto

    @property
    def ultima_curvatura(self) -> float:
        """Curvatura normalizada [0, 1] del último frame. 0=recta, 1=curva cerrada."""
        return self._ultima_curvatura

    # Barrido adaptativo: si la fila primaria no tiene verde, baja en pasos de 40px
    # hasta un máximo de _FILA_MAX. Permite funcionar tanto en autopista (72%)
    # como en ciudad/tráfico (fallback hasta 90%).
    _FILA_MAX = 0.90
    _SWEEP_PX = 40

    def calcular_giro(self, mascara_camino: np.ndarray, ll_mask: np.ndarray | None = None) -> tuple[float, bool]:
        """
        Calcula el error de dirección a partir de la máscara del área manejable.

        Returns:
            (error_norm, carril_perdido)
            error_norm  ∈ [-1, 1]: positivo → girar izquierda, negativo → girar derecha
            carril_perdido: True cuando no hay píxeles de carril visibles
        """
        alto, ancho = mascara_camino.shape
        x_camion = ancho // 2

        curvatura = self._estimar_curvatura(mascara_camino, alto, ancho)
        self._ultima_curvatura = curvatura
        fila_base = int(alto * (self._FILA_LEJOS + curvatura * (self._FILA_CERCA - self._FILA_LEJOS)))

        offsets = [-20, -10,  0, 10, 20]
        pesos   = [ 0.10, 0.20, 0.40, 0.20, 0.10]
        fila_max = int(alto * self._FILA_MAX)

        # Nivel 1a: ajuste polinomial de ll_mask — máxima precisión en curvas
        if ll_mask is not None:
            centro_poly = self._centro_desde_ll_polinomio(ll_mask, fila_base, x_camion, ancho)
            if centro_poly is not None:
                self._ultimo_punto = (int(round(centro_poly)), fila_base)
                dx = x_camion - centro_poly
                error = float(np.clip(dx / (ancho * self._ESCALA_ERROR), -1.0, 1.0))
                error = self._ultimo_error + float(np.clip(
                    error - self._ultimo_error, -self._MAX_DELTA, self._MAX_DELTA
                ))
                self._ultimo_error = error
                return error, False

        # Nivel 1b: ll_mask por filas — fallback cuando píxeles son escasos
        if ll_mask is not None:
            filas_ll = [max(0, min(fila_base + off, alto - 1)) for off in offsets]
            centro_ll = self._centro_desde_ll(ll_mask, filas_ll, x_camion, ancho)
            if centro_ll is not None:
                self._ultimo_punto = (int(round(centro_ll)), fila_base)
                dx = x_camion - centro_ll
                error = float(np.clip(dx / (ancho * self._ESCALA_ERROR), -1.0, 1.0))
                error = self._ultimo_error + float(np.clip(
                    error - self._ultimo_error, -self._MAX_DELTA, self._MAX_DELTA
                ))
                self._ultimo_error = error
                return error, False

        # Nivel 2: centroide da_mask con barrido adaptativo
        # Guardia de obstrucción: pilar de paso inferior, valla o puente bloquean
        # un lado del frame → da_mask unilateral → centroide ficticio → choque.
        if self._obstruccion_detectada(mascara_camino, ancho):
            self._ultimo_punto = None
            self._ultimo_error *= self._DECAY
            return self._ultimo_error, True

        x_sum, w_sum, fila_usada = 0.0, 0.0, fila_base
        fila_try = fila_base
        while fila_try <= fila_max:
            xs, ws = 0.0, 0.0
            for off, peso in zip(offsets, pesos):
                y = max(0, min(fila_try + off, alto - 1))
                x = self._centroide_con_bias(mascara_camino, y, ancho)
                if x is not None:
                    xs += x * peso
                    ws += peso
            if ws > 0.0:
                x_sum, w_sum, fila_usada = xs, ws, fila_try
                break
            fila_try += self._SWEEP_PX

        if w_sum == 0.0:
            self._ultimo_punto = None
            self._ultimo_error *= self._DECAY
            return self._ultimo_error, True

        x_obj = x_sum / w_sum
        self._ultimo_punto = (int(round(x_obj)), fila_usada)

        dx = x_camion - x_obj
        error = float(np.clip(dx / (ancho * self._ESCALA_ERROR), -1.0, 1.0))
        error = self._ultimo_error + float(np.clip(
            error - self._ultimo_error, -self._MAX_DELTA, self._MAX_DELTA
        ))
        self._ultimo_error = error
        return error, False

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _obstruccion_detectada(self, mascara: np.ndarray, ancho: int) -> bool:
        """True cuando da_mask está muy concentrada en un solo lado del frame.

        Un pilar de paso inferior o valla lateral bloquea un lado completo:
        prácticamente cero píxeles izquierda (o derecha) con abundancia en el otro.
        En ese caso el centroide estaría sesgado hasta el extremo → sería peor
        que ignorar la lectura y usar la memoria con decaimiento.

        Distinción clave:
        - Obstrucción real (pilar/puente): AMBOS lados tienen píxeles, pero uno
          tiene muy pocos por reflexión o borde del segmentador. Ratio < 0.08.
        - Borde de carretera: un lado tiene exactamente 0 píxeles (asfalto
          termina). En ese caso el centroide del lado visible es la señal correcta.
        """
        mitad = ancho // 2
        px_izq = int(np.count_nonzero(mascara[:, :mitad]))
        px_der = int(np.count_nonzero(mascara[:, mitad:]))
        total = px_izq + px_der
        if total < self._MIN_PX_SIMETRIA:
            return False   # máscara casi vacía → se trata como carril perdido normal
        menor = min(px_izq, px_der)
        if menor == 0:
            return False   # borde de carretera: centroide del lado visible es válido
        return (menor / total) < self._MIN_SIMETRIA

    def _centroide_con_bias(self, mascara: np.ndarray, fila_y: int, ancho: int) -> int | None:
        """
        Centroide del área manejable en la fila dada (_BIAS_FRAC=0.00 → centroide completo).
        Retorna None si no hay píxeles.
        """
        fila = mascara[fila_y, :]
        indices = np.where(fila > 0)[0]
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
        # Mediana en lugar de media: más robusta al arcén/carril de salida que
        # infla da_mask hacia la derecha, desplazando la media lejos del carril real.
        return int(np.median(indices_der))

    def _estimar_curvatura(self, mascara: np.ndarray, alto: int, ancho: int) -> float:
        """
        Curvatura ∈ [0, 1] basada en la diferencia horizontal entre el
        centroide cercano (fila 85%) y el lejano (fila 72%).
        """
        y_cerca = int(alto * 0.85)
        y_lejos = int(alto * 0.72)

        x_cerca = self._centroide_con_bias(mascara, y_cerca, ancho)
        x_lejos = self._centroide_con_bias(mascara, y_lejos, ancho)

        if x_cerca is None or x_lejos is None:
            return 0.0

        return float(np.clip(abs(x_cerca - x_lejos) / ancho * self._CURVATURA_SCALE, 0.0, 1.0))

    def _centro_desde_ll_polinomio(
        self,
        ll_mask: np.ndarray,
        fila_target: int,
        x_camion: int,
        ancho: int,
    ) -> float | None:
        """
        Ajusta polinomios de grado 2 a TODOS los píxeles de ll_mask en la ROI y
        predice el centro del carril en fila_target.

        Ventaja sobre el enfoque por filas: en una curva pronunciada, la línea
        izquierda o derecha puede estar ausente justo en las filas del look-ahead
        (el camión ya giró) pero visible más abajo. El polinomio extrapola la
        trayectoria correcta usando toda la información disponible.

        El ajuste es x = f(y): usa y como variable independiente para mayor
        estabilidad numérica cuando las líneas son casi verticales.

        Retorna None si:
        - Menos de _MIN_LL_POLY píxeles por lado
        - El ancho de carril predicho está fuera del rango válido
        - El ajuste polinomial falla numéricamente
        """
        alto = ll_mask.shape[0]
        y_inicio = int(alto * self._ROI_POLY_FRAC)

        ys_all, xs_all = np.where(ll_mask[y_inicio:, :] > 0)
        if len(ys_all) < self._MIN_LL_POLY * 2:
            return None
        ys_all = ys_all + y_inicio  # corregir offset de ROI

        izq_mask = xs_all < x_camion
        der_mask = xs_all >= x_camion

        ys_izq, xs_izq = ys_all[izq_mask], xs_all[izq_mask]
        ys_der, xs_der = ys_all[der_mask], xs_all[der_mask]

        if len(xs_izq) < self._MIN_LL_POLY or len(xs_der) < self._MIN_LL_POLY:
            return None

        try:
            poly_izq = np.polyfit(ys_izq, xs_izq, 2)
            poly_der = np.polyfit(ys_der, xs_der, 2)

            x_izq = float(np.polyval(poly_izq, fila_target))
            x_der = float(np.polyval(poly_der, fila_target))

            lane_width = x_der - x_izq
            # En curvas el carril puede parecer más estrecho al mirar de lado;
            # rango ampliado a 55% para no rechazar curvas pronunciadas.
            if not (35 <= lane_width <= int(ancho * 0.55)):
                return None

            return (x_izq + x_der) / 2.0
        except (np.linalg.LinAlgError, ValueError):
            return None

    def _centro_desde_ll(
        self,
        ll_mask: np.ndarray,
        filas: list[int],
        x_camion: int,
        ancho: int,
    ) -> float | None:
        """
        Centro geométrico del carril actual desde ll_mask.

        Acumula píxeles de las filas dadas, los separa en izquierda/derecha
        respecto a x_camion (= frame center) y retorna (max_izq + min_der) / 2.
        En vías de N carriles, max_izq y min_der son siempre los bordes del
        carril actual (los más cercanos al camión), no los de los carriles vecinos.
        Retorna None si algún lado tiene menos de _MIN_LL_PIXELES píxeles.
        """
        pixeles_izq: list[int] = []
        pixeles_der: list[int] = []
        alto = ll_mask.shape[0]
        for fila_y in filas:
            if fila_y < 0 or fila_y >= alto:
                continue
            indices = np.where(ll_mask[fila_y, :] > 0)[0]
            for x in indices:
                if x < x_camion:
                    pixeles_izq.append(int(x))
                else:
                    pixeles_der.append(int(x))
        if len(pixeles_izq) < self._MIN_LL_PIXELES or len(pixeles_der) < self._MIN_LL_PIXELES:
            return None
        borde_izq = int(np.max(pixeles_izq))
        borde_der = int(np.min(pixeles_der))
        # Validar ancho del carril: un carril válido ocupa 3–44 % del ancho del frame
        # en la distancia de anticipación. Un ancho mayor indica que borde_der tomó
        # la línea continua del arcén o carril de salida, no el borde real del carril.
        lane_width = borde_der - borde_izq
        if not (40 <= lane_width <= int(ancho * 0.44)):
            return None
        return (borde_izq + borde_der) / 2.0
