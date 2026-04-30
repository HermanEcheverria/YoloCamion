# tests/test_pure_pursuit.py
import numpy as np
import pytest

from src.control.pure_pursuit import PurePursuitVisual


def test_carril_perdido_devuelve_True_y_error_cero_inicial():
    """Máscara vacía sin memoria previa → (0.0, True)."""
    pp = PurePursuitVisual()
    mascara = np.zeros((480, 640), dtype=np.uint8)
    error, perdido = pp.calcular_giro(mascara)
    assert perdido is True
    assert error == pytest.approx(0.0)


def test_decaimiento_memoria_tras_perder_carril():
    """Tras detectar error, máscara vacía devuelve error_anterior * 0.85."""
    pp = PurePursuitVisual()
    m1 = np.zeros((480, 640), dtype=np.uint8)
    m1[200:480, 50:250] = 1          # área a la izquierda → error positivo
    error_1, perdido_1 = pp.calcular_giro(m1)
    assert not perdido_1
    assert error_1 > 0

    m2 = np.zeros((480, 640), dtype=np.uint8)
    error_2, perdido_2 = pp.calcular_giro(m2)
    assert perdido_2 is True
    assert error_2 == pytest.approx(error_1 * 0.85, rel=0.05)


def test_via_ancha_con_sesgo_derecho_error_negativo():
    """Vía simétrica con _BIAS_FRAC=0.30: centroide desplazado a la derecha → error < 0 (girar derecha).

    _BIAS_FRAC=0.30 modela conducción europea (carril derecho): en vías bidireccionales
    donde da_mask cubre ambos carriles, el centroide de la mitad derecha del área verde
    corresponde al carril del conductor, no a la línea central.
    """
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 0:640] = 1            # área verde simétrica
    error, perdido = pp.calcular_giro(m)
    assert not perdido
    # Con BIAS_FRAC=0.30, centroide queda a la derecha de x_camion → error negativo
    assert error < -0.10


def test_area_solo_izquierda_error_positivo():
    """Área manejable solo a la izquierda → camión debe girar izquierda (error > 0)."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 0:200] = 1
    error, perdido = pp.calcular_giro(m)
    assert not perdido
    assert error > 0.10


def test_area_solo_derecha_error_negativo():
    """Área manejable solo a la derecha → camión debe girar derecha (error < 0)."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 440:640] = 1
    error, perdido = pp.calcular_giro(m)
    assert not perdido
    assert error < -0.10


def test_error_acotado_entre_menos1_y_1():
    """El error normalizado nunca sale del rango [-1, 1]."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 0:10] = 1             # franja extrema → dx muy grande
    error, _ = pp.calcular_giro(m)
    assert -1.0 <= error <= 1.0


def test_ultimo_punto_debug_none_cuando_carril_perdido():
    """Sin carril visible, ultimo_punto_debug debe ser None."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    pp.calcular_giro(m)
    assert pp.ultimo_punto_debug is None


def test_ultimo_punto_debug_dentro_de_la_imagen():
    """Con carril visible, ultimo_punto_debug cae dentro de los límites de la imagen."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 200:500] = 1
    pp.calcular_giro(m)
    assert pp.ultimo_punto_debug is not None
    x, y = pp.ultimo_punto_debug
    assert 0 <= x < 640
    assert 0 <= y < 480


def test_ll_mask_simetrico_error_cero():
    """Líneas equidistantes del centro del frame → centro_carril = x_camion → error ≈ 0."""
    pp = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[300:480, 215:225] = 1   # borde izquierdo  (max = 224)
    ll[300:480, 415:425] = 1   # borde derecho    (min = 415)
    # centro_carril = (224 + 415) / 2 = 319.5 ≈ x_camion=320 → error ≈ 0
    error, perdido = pp.calcular_giro(da, ll)
    assert not perdido
    assert abs(error) < 0.05


def test_ll_mask_corrige_bias_da_mask():
    """
    da_mask asimétrica (centroide ≈ 420 → error negativo).
    ll_mask dice centro ≈ 320 → error ≈ 0.
    """
    pp_con_ll = PurePursuitVisual()
    pp_sin_ll = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[200:480, 200:640] = 1   # centroide ≈ 419 (a la derecha del frame)
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[300:480, 215:225] = 1   # borde izq ≈ 220
    ll[300:480, 415:425] = 1   # borde der ≈ 420 → centro ≈ 320

    error_con, _ = pp_con_ll.calcular_giro(da, ll)
    error_sin, _ = pp_sin_ll.calcular_giro(da)

    assert abs(error_con) < 0.05    # ll_mask: camión centrado
    assert error_sin < -0.10        # da_mask sola: sesgo negativo persistente


def test_ll_mask_vacio_usa_da_mask():
    """ll_mask sin píxeles → resultado idéntico a no pasar ll_mask."""
    pp1 = PurePursuitVisual()
    pp2 = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 50:300] = 1    # área asimétrica
    ll = np.zeros((480, 640), dtype=np.uint8)   # vacía

    e1, _ = pp1.calcular_giro(da)
    e2, _ = pp2.calcular_giro(da, ll)
    assert e1 == pytest.approx(e2, rel=0.01)


def test_ll_mask_pocos_pixeles_no_activa_ll():
    """Menos de _MIN_LL_PIXELES (15) por lado → cae a da_mask, misma señal."""
    pp_ll = PurePursuitVisual()
    pp_da = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1     # da simétrica
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[350, 100] = 1            # 1 pixel izq — insuficiente (< 15)
    ll[350, 540] = 1            # 1 pixel der — insuficiente (< 15)

    e_ll, _ = pp_ll.calcular_giro(da, ll)
    e_da, _ = pp_da.calcular_giro(da)
    assert e_ll == pytest.approx(e_da)


# ── Tests de ajuste polinomial (Nivel 1a) ────────────────────────────────────

def _ll_mask_curva(h: int, w: int, offset_x: int = 0) -> np.ndarray:
    """
    Genera una ll_mask sintética con dos líneas curvadas (parábola) que
    simulan cómo YOLOP devuelve las marcas en una curva a la derecha.

    offset_x desplaza ambas líneas lateralmente para simular descentramiento.
    """
    ll = np.zeros((h, w), dtype=np.uint8)
    x_centro = w // 2 + offset_x
    ancho_carril = int(w * 0.22)   # ~22% del ancho = carril estándar

    for y in range(int(h * 0.60), h):
        # Curvatura simulada: las líneas giran hacia la derecha conforme y sube
        curva = int(0.0008 * (y - h) ** 2)  # parábola abierta hacia abajo
        x_izq = x_centro - ancho_carril // 2 + curva
        x_der = x_centro + ancho_carril // 2 + curva
        for x_l, grosor in [(x_izq, 6), (x_der, 6)]:
            x0 = max(0, x_l - grosor // 2)
            x1 = min(w - 1, x_l + grosor // 2)
            ll[y, x0:x1] = 1
    return ll


def test_polinomio_curva_simetrica_error_pequeno():
    """Líneas curvadas simétricas respecto al camión → error ≈ 0."""
    pp = PurePursuitVisual()
    h, w = 720, 1280
    da = np.ones((h, w), dtype=np.uint8)
    da[:int(h * 0.60), :] = 0
    ll = _ll_mask_curva(h, w, offset_x=0)

    error, perdido = pp.calcular_giro(da, ll)
    assert not perdido
    assert abs(error) < 0.12, f"Error en curva simétrica demasiado grande: {error:.3f}"


def test_polinomio_curva_descentrado_error_correcto():
    """Camión desplazado a la derecha: ambas líneas van a la izquierda → error > 0 (girar izq)."""
    pp = PurePursuitVisual()
    h, w = 720, 1280
    da = np.ones((h, w), dtype=np.uint8)
    da[:int(h * 0.60), :] = 0
    # Desplazar líneas a la izquierda (= camión está a la derecha del carril)
    ll = _ll_mask_curva(h, w, offset_x=-120)

    error, perdido = pp.calcular_giro(da, ll)
    assert not perdido
    # Con desplazamiento negativo, el camión está a la derecha → error positivo (girar izq)
    assert error > 0.10, f"Esperaba error positivo, obtuvo {error:.3f}"


def test_polinomio_activo_antes_que_filas():
    """Con suficientes píxeles en ll_mask, el polinomio debe activarse (Nivel 1a).

    Verificamos que el error con curva sintética difiere del resultado sin ll_mask,
    lo que confirma que el nivel 1a está siendo usado (si fuera da_mask se usaría
    el bias de 0.30 que da errores distintos).
    """
    pp_poly = PurePursuitVisual()
    pp_da   = PurePursuitVisual()
    h, w = 720, 1280
    da = np.ones((h, w), dtype=np.uint8)
    da[:int(h * 0.60), :] = 0
    ll = _ll_mask_curva(h, w, offset_x=0)

    e_poly, perdido_poly = pp_poly.calcular_giro(da, ll)
    e_da,   perdido_da   = pp_da.calcular_giro(da)

    assert not perdido_poly
    assert not perdido_da
    # El polinomio centra en ~0, la da_mask con bias da error < -0.10
    assert abs(e_poly) < abs(e_da), (
        f"Polinomio debería dar error menor que da_mask: poly={e_poly:.3f} da={e_da:.3f}"
    )


def test_ultima_curvatura_sube_en_curva():
    """En una máscara curvada, ultima_curvatura debe ser mayor que en una recta."""
    pp_recta = PurePursuitVisual()
    pp_curva = PurePursuitVisual()

    h, w = 720, 1280

    # Recta: da_mask simétrica sin desplazamiento lateral
    da_recta = np.zeros((h, w), dtype=np.uint8)
    da_recta[int(h*0.60):, int(w*0.35):int(w*0.65)] = 1
    pp_recta.calcular_giro(da_recta)

    # Curva: da_mask desplazada: el centroide lejano y el cercano difieren mucho
    da_curva = np.zeros((h, w), dtype=np.uint8)
    da_curva[int(h*0.60):int(h*0.75), int(w*0.30):int(w*0.60)] = 1   # lejos: izquierda
    da_curva[int(h*0.85):, int(w*0.55):int(w*0.85)] = 1              # cerca: derecha
    pp_curva.calcular_giro(da_curva)

    assert pp_curva.ultima_curvatura > pp_recta.ultima_curvatura, (
        f"curva={pp_curva.ultima_curvatura:.3f} debería ser > recta={pp_recta.ultima_curvatura:.3f}"
    )
