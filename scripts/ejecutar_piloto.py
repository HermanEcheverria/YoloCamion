"""Punto de entrada principal del sistema de conducción autónoma.

Integra los 7 módulos del pipeline:
  fuente → tracker → contexto → FSM → control → registro → seguridad

Uso:
  python scripts/ejecutar_piloto.py                     # usa config/default.yaml
  python scripts/ejecutar_piloto.py --config mi.yaml
  python scripts/ejecutar_piloto.py --control gamepad   # sobreescribe control
  python scripts/ejecutar_piloto.py --fuente pantalla   # captura en vivo
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import yaml
import numpy as np

# Asegurar que src/ está en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.control import ControladorGamepad, ControladorNulo, ControladorTeclado
from src.control.gamepad_pid import ControladorGamepadPID
from src.decision import FSMDecision
from src.fuente import FuentePantalla, FuenteVideo
from src.fuente.buffer import FuenteConBuffer
from src.fuente.ventana import FuenteVentana, buscar_ventana
from src.percepcion import AnalizadorContexto, Tracker
from src.percepcion.contexto import cargar_rois_yaml
from src.percepcion.carriles import DetectorCarriles
from src.percepcion.yolop_inference import InferenciaYOLOP
from src.percepcion.analisis_carriles import AnalizadorCarriles, superponer_carriles
from src.control.pure_pursuit import PurePursuitVisual
from src.percepcion.fisica import EstimadorFisicaVisual
from src.percepcion.velocidad_dashboard import EstimadorVelocidadDashboard
from src.registro import GrabadorVideo, LoggerJSONL, MetricasSesion
from src.seguridad import MonitorSeguridad
from src.tipos import Accion, ComandoControl, SetpointControl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("piloto")

# Acciones con giro intencional: el FSM ya fija desviacion_volante segun la
# accion y el detector de carriles NO debe sobreescribirla (es maniobra activa,
# no seguimiento de carril).
_ACCIONES_CON_GIRO: frozenset[Accion] = frozenset({
    Accion.GIRAR_IZQ, Accion.GIRAR_DER,
    Accion.REBASAR_IZQ, Accion.REBASAR_DER,
})

_MIN_PIXELES_LL_YOLOP = 800
_MIN_PIXELES_LL_LADO_YOLOP = 250
_ZONA_MUERTA_CARRIL_PP = 0.015
_GANANCIA_CARRIL_PP = 1.45
_LIMITE_COMANDO_DA = 0.25


def _setpoint_a_comando(sp: SetpointControl) -> ComandoControl:
    """Adaptador: SetpointControl -> ComandoControl para controladores no-PID.

    desviacion_volante ya viene como stick command (ganancia e inversión aplicadas).
    """
    return ComandoControl(
        acelerador=sp.velocidad_objetivo_norm,
        freno=sp.freno_objetivo,
        volante=sp.desviacion_volante,
        timestamp=time.monotonic(),
    )


def _ll_yolop_valida(ll_mask_roi: np.ndarray) -> tuple[bool, int, int, int]:
    """Valida que YOLOP vea marcas suficientes a ambos lados del camion."""
    alto, ancho = ll_mask_roi.shape
    total_full = int(np.count_nonzero(ll_mask_roi))
    y0 = int(round(alto * 0.66))
    y1 = int(round(alto * 0.92))
    x_ref = ancho // 2
    half = int(round(ancho * 0.24))
    x0 = max(0, x_ref - half)
    x2 = min(ancho, x_ref + half)

    roi = ll_mask_roi[y0:y1, x0:x2]
    xs = np.nonzero(roi)[1]
    total = int(xs.size)
    if total == 0:
        return False, total_full, 0, 0

    x_split = x_ref - x0
    pix_izq = int(np.count_nonzero(xs < x_split))
    pix_der = int(np.count_nonzero(xs >= x_split))
    valida = (
        total >= _MIN_PIXELES_LL_YOLOP
        and pix_izq >= _MIN_PIXELES_LL_LADO_YOLOP
        and pix_der >= _MIN_PIXELES_LL_LADO_YOLOP
    )
    return valida, total_full, pix_izq, pix_der


def cargar_config(ruta: str) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def construir_fuente(cfg: dict):
    tipo = cfg["fuente"]["tipo"]
    if tipo == "video":
        return FuenteVideo(cfg["fuente"]["ruta_video"])
    elif tipo == "pantalla":
        escalar = cfg["fuente"].get("escalar_a")
        pantalla = FuentePantalla(
            monitor=cfg["fuente"].get("monitor", 0),
            region=cfg["fuente"].get("region"),
            escalar_a=tuple(escalar) if escalar else None,
        )
        return FuenteConBuffer(pantalla)
    elif tipo == "ventana":
        titulo = cfg["fuente"].get("titulo_ventana", "Euro Truck Simulator 2")
        escalar = cfg["fuente"].get("escalar_a", [1920, 1080])
        ventana = FuenteVentana(
            titulo=titulo,
            escalar_a=tuple(escalar) if escalar else None,
        )
        return FuenteConBuffer(ventana)
    raise ValueError(f"Tipo de fuente desconocido: {tipo}")


def construir_controlador(cfg: dict, tipo_override: str | None = None):
    tipo = tipo_override or cfg["control"]["tipo"]
    if tipo == "nulo":
        return ControladorNulo()
    elif tipo == "gamepad":
        # Pure-vision: gamepad analogico con tres PIDs (Tarea 3.2-3.4)
        ctrl = ControladorGamepadPID()
        ctrl.iniciar()
        return ctrl
    elif tipo == "gamepad_directo":
        # Fallback sin PID: pasthrough analogico (no recomendado)
        ctrl = ControladorGamepad()
        ctrl.iniciar()
        return ctrl
    elif tipo == "teclado":
        return ControladorTeclado()
    raise ValueError(f"Tipo de control desconocido: {tipo}")


def countdown(segundos: int) -> None:
    """Cuenta regresiva visible en consola para que el usuario cambie al juego."""
    print("\n" + "="*50)
    print("  Cambia al juego ETS2 AHORA")
    print("  El piloto arrancará en:")
    for i in range(segundos, 0, -1):
        print(f"    {i}...", flush=True)
        time.sleep(1)
    print("  ¡INICIANDO!\n" + "="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Piloto autónomo ETS2")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--control", default=None, help="Sobreescribir tipo de control")
    parser.add_argument("--fuente", default=None, help="Sobreescribir tipo de fuente")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = sin límite")
    parser.add_argument("--delay", type=int, default=0,
                        help="Segundos de countdown antes de arrancar (útil para cambiar al juego)")
    parser.add_argument("--sin-video", action="store_true",
                        help="No grabar video (más rápido, recomendado para pruebas en vivo)")
    parser.add_argument("--debug-carril", action="store_true",
                        help="Mostrar estado del carril cada 30 frames para calibración")
    parser.add_argument("--debug-carril-img", action="store_true",
                        help="Guardar imagen de debug con líneas detectadas cada 60 frames")
    parser.add_argument("--debug-yolop", action="store_true",
                        help="Guardar imagen compuesta cada 60 frames: entrada del modelo | máscaras superpuestas")
    parser.add_argument("--debug-clasif-carriles", action="store_true",
                        help="Guardar imagen con clasificación de carriles (ego/contrario/mismo) cada 60 frames")
    parser.add_argument("--debug-vel", action="store_true",
                        help="Guardar el ROI del OCR de velocidad cada 30 frames para calibrar _ROI_DIGITOS")
    parser.add_argument("--mostrar", action="store_true",
                        help="Mostrar ventana en vivo con detecciones YOLO y estado FSM (reduce ~3 FPS)")
    args = parser.parse_args()

    cfg = cargar_config(args.config)

    # Sobreescrituras de CLI
    if args.control:
        cfg["control"]["tipo"] = args.control
    if args.fuente:
        cfg["fuente"]["tipo"] = args.fuente

    logger.info("=== Iniciando piloto autónomo ETS2 ===")
    logger.info("Fuente: %s | Control: %s", cfg["fuente"]["tipo"], cfg["control"]["tipo"])

    if args.delay > 0:
        countdown(args.delay)

    # Cargar ROI calibradas
    ruta_rois = Path("config/regiones_interes.yaml")
    rois = cargar_rois_yaml(ruta_rois) if ruta_rois.exists() else None
    if rois:
        logger.info("ROI cargadas desde %s (%d regiones)", ruta_rois, len(rois))
    else:
        logger.warning("No se encontró %s — usando ROI por defecto", ruta_rois)

    # Construir componentes
    fuente = construir_fuente(cfg)
    tracker = Tracker(
        ruta_modelo=cfg["modelo"]["pesos"],
        confianza_min=cfg["modelo"]["conf_min"],
        imgsz=cfg["modelo"]["imgsz"],
        device=cfg["modelo"]["device"],
    )
    estimador_fisica = EstimadorFisicaVisual()
    contexto = AnalizadorContexto(rois=rois, estimador_fisica=estimador_fisica)
    fsm = FSMDecision()
    yolop = InferenciaYOLOP(
        imgsz=cfg["modelo"]["imgsz"],
        device=cfg["modelo"]["device"],
    )
    analizador_carriles = AnalizadorCarriles(usar_suavizado=True)
    pure_pursuit = PurePursuitVisual()
    controlador = construir_controlador(cfg)

    # Velocidad propia desde el HUD. Es mas fiable que flujo optico para detectar
    # 0 km/h; en ETS2, aplicar LT parado engrana reversa.
    cfg_vel_dash = cfg.get("velocidad_dashboard", {}) or {}
    estimador_velocidad = EstimadorVelocidadDashboard(
        max_kmh_norm=float(cfg_vel_dash.get("max_kmh_norm", 90.0)),
        retener_frames=int(cfg_vel_dash.get("retener_frames", 15)),
    )
    velocidad_actual_norm = 0.0
    velocidad_actual_kmh: int | None = None
    _vel_estimada: float = 0.0  # estimación suavizada; se usa solo para logging y condiciones de curva
    metricas = MetricasSesion()
    log = LoggerJSONL(cfg["registro"]["ruta_base"])
    grabar = cfg["registro"]["grabar_video"] and not args.sin_video
    grabador = GrabadorVideo(cfg["registro"]["ruta_base"]) if grabar else None
    if args.sin_video:
        logger.info("Grabación de video desactivada (--sin-video)")

    def en_paro():
        fsm.activar_paro_manual()
        controlador.liberar()
        log.seguridad("paro de emergencia activado")
        metricas.registrar_evento_seguridad()

    monitor = MonitorSeguridad(
        en_paro=en_paro,
        tecla_paro=cfg["seguridad"]["tecla_paro"],
        timeout_ms=cfg["seguridad"]["timeout_watchdog_ms"],
    )

    try:
        logger.info("Cargando modelos YOLO...")
        tracker.cargar()
        yolop.cargar()

        # Warmup: pre-compila kernels CUDA para que el primer frame real sea rápido.
        # Se ejecutan 3 pasadas para asegurar que todos los caminos CUDA están compilados.
        logger.info("Warmup YOLO (compilando kernels CUDA — puede tardar 60s en arranque en frío)...")
        import numpy as _np
        _frame_dummy = _np.zeros((1080, 1920, 3), dtype=_np.uint8)
        for _ in range(3):
            tracker.rastrear(_frame_dummy)
            _, _da, _ll = yolop.procesar_frame(_frame_dummy)
            _pp = PurePursuitVisual()
            _pp.calcular_giro(_da, _ll)
        logger.info("Warmup completado — CUDA listo")

        fuente.iniciar()
        monitor.iniciar()

        primer_frame = True
        estado_anterior = fsm.estado_actual
        n_frame = 0
        seguimientos = []    # se actualiza cada YOLO_CADA frames

        # Control bang-bang sobre OCR: acelera si kmh < objetivo, frena si kmh > objetivo.
        # Si el OCR no tiene lectura válida → gas suave constante (_GAS_CRUCERO).
        _VEL_OBJETIVO_KMH = 30    # km/h de crucero para ciudad (reducido de 35)
        _BANDA_KMH        = 4     # histéresis ±2 km/h (gas <28, freno >32, coast entre)
        _GAS_CRUCERO      = 0.14  # 14% RT (~36/255) — físicamente limita a ~30 km/h sin OCR
        _FRENO_CRUCERO    = 0.22  # 22% LT (~56/255) para frenar suave
        _EMERGENCIA_KMH   = 45    # freno de emergencia si OCR lee claramente > 45
        # Cache del último resultado YOLO/FSM — se actualiza cada YOLO_CADA frames
        YOLO_CADA = 3        # YOLO cada 3 frames → ~10 FPS detección, ~30 FPS carril
        yolo_contador = 0
        resultado_cache = None
        # Cache de YOLOP — corre cada 2 frames para reducir latencia (~5 Hz → ~15 Hz)
        YOLOP_CADA = 2
        yolop_contador_carril = 0
        da_mask_cache: np.ndarray | None = None
        ll_mask_cache: np.ndarray | None = None

        # EMA de la desviación lateral del carril.
        # EMA rapida: con gamepad sin deadzone fisica, el piloto puede corregir
        # temprano; demasiada memoria retrasa la salida hasta estar cerca del muro.
        _ALPHA_EMA_CARRIL = 0.45
        desv_ema: float = 0.0

        from src.decision.estado import EstadoFSM
        _ESTADOS_CARRIL = (
            EstadoFSM.CONDUCIENDO_NORMAL,
            EstadoFSM.SIGUIENDO_VEHICULO,
            EstadoFSM.FRENANDO_PREVENTIVO,
            EstadoFSM.APROXIMANDO_ALTO,
            EstadoFSM.APROXIMANDO_SEMAFORO,
            EstadoFSM.RECUPERACION,
        )

        logger.info("Pipeline iniciado — presiona %s para parar", cfg["seguridad"]["tecla_paro"].upper())
        logger.info("YOLO cada %d frames | Carril cada frame", YOLO_CADA)

        while fuente.esta_activa:
            if monitor.paro_activado():
                break
            if args.max_frames > 0 and n_frame >= args.max_frames:
                logger.info("Límite de frames alcanzado (%d)", args.max_frames)
                break

            monitor.heartbeat()
            t0 = time.perf_counter()

            cuadro = fuente.siguiente()
            if cuadro is None:
                if not fuente.esta_activa:
                    break
                time.sleep(0.005)
                continue

            # ── Detección de carril (cada YOLOP_CADA frames — inferencia ~100ms) ──
            yolop_contador_carril += 1
            if yolop_contador_carril >= YOLOP_CADA or da_mask_cache is None:
                yolop_contador_carril = 0
                _, da_mask_cache, ll_mask_cache = yolop.procesar_frame(cuadro.imagen)
            da_mask = da_mask_cache.copy()
            ll_mask = ll_mask_cache

            # Enmascarar zona superior (60%): espejos virtuales ocupan hasta y≈55%.
            # Margen extra de 5% evita que variaciones de ángulo/resolución los expongan.
            # La carretera útil está en el 65-85% de la imagen.
            _fila_roi = int(da_mask.shape[0] * 0.60)
            da_mask[:_fila_roi, :] = 0
            ll_mask_roi = ll_mask.copy()
            ll_mask_roi[:_fila_roi, :] = 0
            ll_yolop_valida, pixeles_ll_yolop, pixeles_ll_izq, pixeles_ll_der = _ll_yolop_valida(ll_mask_roi)

            # Clasificación de carriles (ego / contrario / mismo sentido).
            # Disponible para futuras decisiones del FSM o el control;
            # de momento solo se visualiza con --debug-clasif-carriles.
            carriles_clasif = analizador_carriles.analizar(ll_mask, da_mask)

            # Pure Pursuit: ll_mask (nivel 1) → da_mask centroide (nivel 2) → decay (nivel 3).
            # No usamos el detector clásico de brillo como respaldo: su ROI no tiene en cuenta
            # el offset de cámara (_BIAS_CAM_PX=80), lo que produce un sesgo sistemático a la
            # derecha que hace que el camión se estrelle contra la barrera derecha.
            giro_pure_pursuit, carril_perdido = pure_pursuit.calcular_giro(da_mask, ll_mask_roi)
            fuente_carril = "ll" if ll_yolop_valida else ("da" if not carril_perdido else "decay")
            detalle_carril = ""
            comando_carril_directo: float | None = None

            # EMA de suavizado rapido: PurePursuit ya limita saltos grandes.
            desv_ema = (_ALPHA_EMA_CARRIL * giro_pure_pursuit + (1.0 - _ALPHA_EMA_CARRIL) * desv_ema)

            # Decay del EMA mientras el camión está detenido para evitar que la
            # memoria de desviación cause un sobreimpulso de volante al arrancar.
            # Estados de paro completo: la EMA decae rápido (×0.85 por frame).
            # A baja velocidad (<5 km/h): decay a la mitad por frame.
            _estado_fsm_actual = resultado_cache.estado_nuevo if resultado_cache else None
            _estados_paro = (EstadoFSM.DETENIDO_SEMAFORO, EstadoFSM.DETENIDO_ALTO,
                             EstadoFSM.PARO_EMERGENCIA)
            if _estado_fsm_actual in _estados_paro:
                desv_ema *= 0.85
            elif velocidad_actual_kmh is not None and velocidad_actual_kmh <= 5:
                desv_ema *= 0.5

            # ── Velocidad propia desde HUD (solo para logging) ───────────────
            lectura_velocidad = estimador_velocidad.estimar(cuadro.imagen)
            velocidad_actual_kmh = lectura_velocidad.kmh
            if lectura_velocidad.norm > 0.0:
                _vel_estimada = lectura_velocidad.norm
            else:
                _vel_estimada = max(0.0, _vel_estimada - 0.001)
            velocidad_actual_norm = _vel_estimada

            if args.debug_vel and n_frame % 30 == 0:
                import cv2 as _cv2
                from pathlib import Path as _Path
                roi_dbg = estimador_velocidad._ultimo_roi_debug
                if roi_dbg is not None:
                    # ROI estrecho ampliado ×4 para ver los dígitos exactos
                    roi_grande = _cv2.resize(roi_dbg, (roi_dbg.shape[1] * 4, roi_dbg.shape[0] * 4),
                                             interpolation=_cv2.INTER_NEAREST)
                    ruta_vel = _Path(cfg["registro"]["ruta_base"]) / f"debug_vel_roi_{n_frame:06d}.jpg"
                    _cv2.imwrite(str(ruta_vel), roi_grande)
                    # Imagen de contexto: esquina inferior-izquierda (primeros 20% ancho, últimos 25% alto)
                    # para ver todo el velocímetro y calibrar el ROI
                    h_f, w_f = cuadro.imagen.shape[:2]
                    ctx = cuadro.imagen[int(h_f * 0.75):, :int(w_f * 0.20)].copy()
                    # Dibujar el ROI actual en rojo sobre el contexto
                    from src.percepcion.velocidad_dashboard import _ROI_DIGITOS as _roi
                    rx1 = int(round(w_f * _roi[0])) - 0  # ya en coords absolutas
                    ry1 = int(round(h_f * _roi[1])) - int(h_f * 0.75)
                    rx2 = int(round(w_f * _roi[2]))
                    ry2 = int(round(h_f * _roi[3])) - int(h_f * 0.75)
                    _cv2.rectangle(ctx, (rx1, max(0, ry1)), (min(ctx.shape[1]-1, rx2), max(0, ry2)), (0, 0, 255), 2)
                    _cv2.putText(ctx, f"kmh={velocidad_actual_kmh if velocidad_actual_kmh is not None else '-'} c={lectura_velocidad.confianza:.2f}",
                                 (4, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                    ruta_ctx = _Path(cfg["registro"]["ruta_base"]) / f"debug_vel_ctx_{n_frame:06d}.jpg"
                    _cv2.imwrite(str(ruta_ctx), ctx)
                    logger.info("OCR ROI guardado: %s | ctx: %s | lectura kmh=%s conf=%.2f",
                                ruta_vel, ruta_ctx,
                                "-" if velocidad_actual_kmh is None else str(velocidad_actual_kmh),
                                lectura_velocidad.confianza)

            # ── YOLO + FSM (cada YOLO_CADA frames — lenta ~100ms) ───────────
            yolo_contador += 1
            if yolo_contador >= YOLO_CADA or resultado_cache is None:
                yolo_contador = 0
                seguimientos = tracker.rastrear(cuadro.imagen)
                escena = contexto.analizar(seguimientos, cuadro.imagen, cuadro.timestamp)
                resultado_cache = fsm.decidir(escena)

            resultado = resultado_cache

            # ── Setpoint del FSM (mutable) + override de carril ──────────────
            setpoint = SetpointControl(
                velocidad_objetivo_norm=resultado.setpoint.velocidad_objetivo_norm,
                freno_objetivo=resultado.setpoint.freno_objetivo,
                desviacion_volante=resultado.setpoint.desviacion_volante,
            )

            # Override de carril: activo mientras el camion sigue avanzando.
            # Incluso al aproximarse a alto/semaforo debe mantenerse centrado;
            # solo los estados detenidos dejan el volante al FSM.
            # Zona muerta pequena: corrige deriva antes de acercarse a la linea.
            if (resultado.accion not in _ACCIONES_CON_GIRO
                    and resultado.estado_nuevo in _ESTADOS_CARRIL):
                if comando_carril_directo is not None:
                    setpoint.desviacion_volante = comando_carril_directo
                else:
                    # Error crudo del carril (positivo = necesita girar izq).
                    desv_raw = 0.0 if abs(desv_ema) < _ZONA_MUERTA_CARRIL_PP else float(desv_ema)
                    if fuente_carril == "da":
                        desv_raw = float(np.clip(desv_raw, -_LIMITE_COMANDO_DA, _LIMITE_COMANDO_DA))
                    # Stick command para controladores no-PID (ganancia + inversión de signo).
                    setpoint.desviacion_volante = float(
                        np.clip(-desv_raw * _GANANCIA_CARRIL_PP, -1.0, 1.0)
                    )
                    # Error crudo para _pid_vol en ControladorGamepadPID.
                    setpoint.error_carril = desv_raw

            # Reducir velocidad cuando el carril se pierde completamente.
            if carril_perdido and resultado.estado_nuevo in _ESTADOS_CARRIL:
                setpoint.velocidad_objetivo_norm *= 0.50
                setpoint.freno_objetivo = max(setpoint.freno_objetivo, 0.10)

            # Reducir velocidad en curvas (sin depender del OCR).
            if resultado.estado_nuevo in _ESTADOS_CARRIL:
                curva = max(abs(giro_pure_pursuit), abs(desv_ema), pure_pursuit.ultima_curvatura_debug)
                if curva > 0.06:
                    escala_curva = float(np.interp(curva, [0.06, 0.45], [0.90, 0.40]))
                    setpoint.velocidad_objetivo_norm *= escala_curva
                if curva > 0.12:
                    freno_curva = 0.06 if curva < 0.25 else 0.10
                    setpoint.freno_objetivo = max(setpoint.freno_objetivo, freno_curva)

            # Bang-bang sobre OCR: acelera/frena según kmh vs objetivo.
            _fsm_frena = resultado.accion in {
                Accion.FRENAR_SUAVE, Accion.FRENAR_FUERTE, Accion.ALTO_TOTAL,
            }

            _ocr_emergencia = velocidad_actual_kmh is not None and velocidad_actual_kmh > _EMERGENCIA_KMH
            if _ocr_emergencia:
                setpoint.velocidad_objetivo_norm = 0.0
                setpoint.freno_objetivo = max(setpoint.freno_objetivo, 0.90)
            elif resultado.estado_nuevo in _ESTADOS_CARRIL and not _fsm_frena:
                if velocidad_actual_kmh is None:
                    # OCR sin lectura → gas suave hasta que el OCR reporte
                    setpoint.velocidad_objetivo_norm = _GAS_CRUCERO
                    setpoint.freno_objetivo = 0.0
                elif velocidad_actual_kmh < _VEL_OBJETIVO_KMH - _BANDA_KMH // 2:
                    # Por debajo del objetivo → acelerar
                    setpoint.velocidad_objetivo_norm = _GAS_CRUCERO
                    setpoint.freno_objetivo = 0.0
                elif velocidad_actual_kmh > _VEL_OBJETIVO_KMH + _BANDA_KMH // 2:
                    # Por encima del objetivo → frenar suave
                    setpoint.velocidad_objetivo_norm = 0.0
                    setpoint.freno_objetivo = _FRENO_CRUCERO
                # else: dentro de la banda (33–37 km/h) → coast, sin aplicar nada

            if args.debug_carril and n_frame % 30 == 0:
                logger.info(
                    "CARRIL (%s%s) desv_pp=%+.3f ema=%+.3f curv=%.2f stick_obj=%+.2f vel=%.2f kmh=%s ll_px=%d/%d/%d",
                    fuente_carril,
                    detalle_carril,
                    giro_pure_pursuit, desv_ema,
                    pure_pursuit.ultima_curvatura_debug,
                    setpoint.desviacion_volante, velocidad_actual_norm,
                    "-" if velocidad_actual_kmh is None else str(velocidad_actual_kmh),
                    pixeles_ll_yolop, pixeles_ll_izq, pixeles_ll_der,
                )

            if args.debug_carril_img and n_frame % 60 == 0:
                import cv2 as _cv2
                from pathlib import Path as _Path
                # Crear imagen de debug combinando mascara verde sobre el frame
                dbg = cuadro.imagen.copy()
                mask_color = np.zeros_like(dbg)
                mask_color[da_mask > 0] = (0, 255, 0) # Verde para drivable area
                mask_color[ll_mask > 0] = (0, 0, 255) # Rojo para lane lines
                dbg = _cv2.addWeighted(dbg, 0.7, mask_color, 0.3, 0)

                # Dibujar look-ahead point
                punto = pure_pursuit.ultimo_punto_debug
                if punto:
                    _cv2.circle(dbg, punto, 10, (255, 255, 0), -1)

                ruta_dbg = _Path(cfg["registro"]["ruta_base"]) / f"debug_yolop_{n_frame:06d}.jpg"
                _cv2.imwrite(str(ruta_dbg), dbg)
                logger.info("Debug imagen YOLOP guardada: %s", ruta_dbg)

            if args.debug_yolop and n_frame % 60 == 0:
                import cv2 as _cv2
                from pathlib import Path as _Path
                _H = 486  # altura del panel; ancho proporcional a 16:9

                # Panel izquierdo: frame original capturado (lo que el piloto ve)
                orig = _cv2.resize(cuadro.imagen, (_H * 16 // 9, _H))
                _cv2.putText(orig, "CAPTURA ORIGINAL", (8, 22),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Panel central: entrada real del modelo (CLAHE + sharpened + letterbox)
                entrada = yolop.ultima_imagen_debug
                if entrada is not None:
                    # Escalar manteniendo el cuadrado letterbox visible
                    entrada = _cv2.resize(entrada, (_H, _H))
                    _cv2.putText(entrada, "ENTRADA MODELO (CLAHE+sharp+LB)", (8, 22),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    entrada = np.zeros((_H, _H, 3), dtype=np.uint8)

                # Panel derecho: máscaras + look-ahead sobre frame original
                mascaras = cuadro.imagen.copy()
                mc = np.zeros_like(mascaras)
                mc[da_mask > 0] = (0, 200, 0)   # verde = drivable area
                mc[ll_mask > 0] = (0, 80, 255)  # naranja-rojo = lane lines
                mascaras = _cv2.addWeighted(mascaras, 0.65, mc, 0.35, 0)
                punto = pure_pursuit.ultimo_punto_debug
                if punto:
                    _cv2.circle(mascaras, punto, 12, (0, 255, 255), -1)
                    _cv2.circle(mascaras, punto, 12, (0, 0, 0), 2)
                # Anotar error de carril actual
                _cv2.putText(mascaras,
                             f"src={fuente_carril} ll={pixeles_ll_yolop}/{pixeles_ll_izq}/{pixeles_ll_der} err={giro_pure_pursuit:+.3f} ema={desv_ema:+.3f} cmd={setpoint.desviacion_volante:+.3f} kmh={velocidad_actual_kmh if velocidad_actual_kmh is not None else '-'} curv={pure_pursuit.ultima_curvatura_debug:.2f}{detalle_carril}",
                             (8, 22), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                _cv2.putText(mascaras, "MASCARAS + LOOK-AHEAD", (8, 48),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                mascaras = _cv2.resize(mascaras, (_H * 16 // 9, _H))

                # Separadores verticales de 4px
                sep = np.full((_H, 4, 3), 60, dtype=np.uint8)
                compuesto = np.hstack([orig, sep, entrada, sep, mascaras])

                ruta_cmp = _Path(cfg["registro"]["ruta_base"]) / f"debug_modelo_{n_frame:06d}.jpg"
                _cv2.imwrite(str(ruta_cmp), compuesto, [_cv2.IMWRITE_JPEG_QUALITY, 90])
                logger.info("Debug modelo guardado: %s", ruta_cmp)

            if args.debug_clasif_carriles and n_frame % 60 == 0:
                import cv2 as _cv2
                from pathlib import Path as _Path
                clasif_img = superponer_carriles(
                    cuadro.imagen,
                    carriles_clasif,
                    area_mask=da_mask,
                    fps=cuadro.fps_instantaneo,
                    frame_idx=n_frame,
                )
                ruta_clasif = _Path(cfg["registro"]["ruta_base"]) / f"debug_carriles_{n_frame:06d}.jpg"
                _cv2.imwrite(str(ruta_clasif), clasif_img, [_cv2.IMWRITE_JPEG_QUALITY, 90])
                logger.info(
                    "Debug clasificación carriles guardado: %s | estado=%s offset=%s",
                    ruta_clasif, carriles_clasif.estado,
                    f"{carriles_clasif.offset_px:+.0f}px" if carriles_clasif.offset_px is not None else "-",
                )

            # ── Ventana de debug en vivo (--mostrar) ────────────────────────
            # La ventana se configura como siempre-encima y sin robar foco
            # (WS_EX_NOACTIVATE) para que el juego mantenga el control del gamepad
            # al hacer clic sobre ella.
            if args.mostrar:
                import cv2 as _cv2
                _canvas = cuadro.imagen.copy()
                _color_caja = {
                    "vehiculo": (0, 165, 255), "motocicleta": (0, 165, 255),
                    "peaton": (0, 0, 255), "semaforo": (255, 255, 0),
                    "senal_alto": (0, 255, 255), "desconocido": (128, 128, 128),
                }
                _color_accion_disp = {
                    Accion.ALTO_TOTAL: (0, 0, 255), Accion.FRENAR_FUERTE: (0, 0, 200),
                    Accion.FRENAR_SUAVE: (0, 165, 255), Accion.MANTENER: (255, 255, 255),
                    Accion.ACELERAR: (0, 255, 0),
                }
                for _seg in seguimientos:
                    _x1, _y1, _x2, _y2 = _seg.caja
                    _col = _color_caja.get(_seg.clase.value, (128, 128, 128))
                    _cv2.rectangle(_canvas, (_x1, _y1), (_x2, _y2), _col, 2)
                    _ttc_str = ""
                    if _seg.fisica and _seg.fisica.ttc_segundos < 10:
                        _ttc_str = f" TTC={_seg.fisica.ttc_segundos:.1f}s"
                    _lbl = f"{_seg.clase.value}#{_seg.id_seguimiento}{_ttc_str}"
                    _cv2.putText(_canvas, _lbl, (_x1, max(_y1 - 5, 12)),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.45, _col, 1, _cv2.LINE_AA)
                _col_a = _color_accion_disp.get(resultado.accion, (255, 255, 255))
                _cv2.rectangle(_canvas, (0, 0), (500, 105), (0, 0, 0), -1)
                _cv2.putText(_canvas, f"Accion: {resultado.accion.value}", (8, 28),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.85, _col_a, 2, _cv2.LINE_AA)
                _cv2.putText(_canvas, f"Estado: {resultado.estado_nuevo.value}  R{resultado.regla}", (8, 56),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, _cv2.LINE_AA)
                _cv2.putText(_canvas, f"R{resultado.regla}: {resultado.razon[:65]}", (8, 80),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, _cv2.LINE_AA)
                _cv2.putText(_canvas,
                             f"kmh={velocidad_actual_kmh if velocidad_actual_kmh is not None else '-'}"
                             f"  FPS={cuadro.fps_instantaneo:.0f}  frm={n_frame}",
                             (8, 100), _cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 255, 150), 1, _cv2.LINE_AA)
                _fh, _fw = _canvas.shape[:2]
                # Escalar a ancho fijo independiente de la resolución capturada
                _dbg_w = 960
                _dbg_h = int(_fh * _dbg_w / _fw)
                _cv2.imshow("YOLO Debug", _cv2.resize(_canvas, (_dbg_w, _dbg_h)))
                _cv2.waitKey(1)

                # Primera vez: configurar ventana siempre-encima sin robo de foco
                if n_frame == 1:
                    import ctypes as _ct
                    import ctypes.wintypes as _wt
                    _hwnd = _ct.windll.user32.FindWindowW(None, "YOLO Debug")
                    if _hwnd:
                        # WS_EX_NOACTIVATE: clic no roba foco del juego
                        _GWL_EXSTYLE = -20
                        _WS_EX_NOACTIVATE = 0x08000000
                        _st = _ct.windll.user32.GetWindowLongW(_hwnd, _GWL_EXSTYLE)
                        _ct.windll.user32.SetWindowLongW(_hwnd, _GWL_EXSTYLE, _st | _WS_EX_NOACTIVATE)

                        # Localizar ETS2 para saber dónde posicionar la ventana de debug
                        _pos_x, _pos_y = 5, 35
                        _hwnd_ets = _ct.windll.user32.FindWindowW(None, "Euro Truck Simulator 2")
                        if _hwnd_ets:
                            _rect = _wt.RECT()
                            _ct.windll.user32.GetWindowRect(_hwnd_ets, _ct.byref(_rect))
                            _pos_x = _rect.left + 5
                            _pos_y = _rect.top + 35  # debajo de la barra de título de ETS2
                            # ETS2 en modo ventana suele marcarse topmost; quitarle ese flag
                            # para que nuestra ventana de debug pueda quedar encima.
                            # HWND_NOTOPMOST=-2, SWP_NOMOVE|SWP_NOSIZE|SWP_NOACTIVATE
                            _ct.windll.user32.SetWindowPos(_hwnd_ets, -2, 0, 0, 0, 0, 0x0013)

                        # Poner nuestra ventana encima de todo: HWND_TOPMOST=-1
                        _ct.windll.user32.SetWindowPos(_hwnd, -1, _pos_x, _pos_y, _dbg_w, _dbg_h, 0x0010)
                        logger.info(
                            "Ventana YOLO Debug: %dx%d en (%d,%d)",
                            _dbg_w, _dbg_h, _pos_x, _pos_y
                        )

            if isinstance(controlador, ControladorGamepadPID):
                controlador.aplicar(setpoint)
                if args.debug_carril and n_frame % 30 == 0:
                    rt_aplicado, lt_aplicado, stick_aplicado = controlador.ultimo_comando_aplicado
                    logger.info(
                        "GAMEPAD aplicado rt=%d lt=%d stick=%+.2f",
                        rt_aplicado, lt_aplicado, stick_aplicado,
                    )
                    log.evento("carril_control", {
                        "frame": n_frame,
                        "fuente": fuente_carril,
                        "detalle": detalle_carril.strip(),
                        "err": round(float(giro_pure_pursuit), 4),
                        "ema": round(float(desv_ema), 4),
                        "err_pid": round(float(setpoint.error_carril or 0.0), 4),
                        "cmd": round(float(setpoint.desviacion_volante), 4),
                        "stick": round(float(stick_aplicado), 4),
                        "kmh": velocidad_actual_kmh,
                        "rt": int(rt_aplicado),
                        "lt": int(lt_aplicado),
                        "ll_total": int(pixeles_ll_yolop),
                        "ll_izq": int(pixeles_ll_izq),
                        "ll_der": int(pixeles_ll_der),
                        "perdido": bool(carril_perdido),
                    })
            else:
                controlador.aplicar(_setpoint_a_comando(setpoint))

            # ── Registro ────────────────────────────────────────────────────
            latencia_ms = (time.perf_counter() - t0) * 1000
            metricas.registrar_frame(cuadro.fps_instantaneo, latencia_ms)
            log.frame(cuadro.indice, cuadro.fps_instantaneo)
            log.decision(resultado.regla, resultado.accion.value,
                         resultado.estado_nuevo.value, resultado.razon)

            if resultado.estado_nuevo != estado_anterior:
                log.transicion(estado_anterior.value, resultado.estado_nuevo.value, resultado.regla)
                logger.info("[R%d] %s → %s | %s",
                            resultado.regla, estado_anterior.value,
                            resultado.estado_nuevo.value, resultado.razon)
                estado_anterior = resultado.estado_nuevo

            if grabador is not None:
                if primer_frame:
                    h, w = cuadro.imagen.shape[:2]
                    grabador.iniciar(w, h)
                    primer_frame = False
                grabador.escribir_frame(
                    cuadro.imagen, seguimientos, resultado.accion,
                    resultado.estado_nuevo.value, cuadro.fps_instantaneo, resultado.regla,
                )

            n_frame += 1

    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario (Ctrl+C)")
    finally:
        monitor.detener()
        controlador.liberar()
        controlador.cerrar()
        fuente.cerrar()
        if grabador:
            grabador.cerrar()
        log.cerrar()
        if args.mostrar:
            import cv2 as _cv2
            _cv2.destroyAllWindows()

        resumen = metricas.resumen()
        logger.info("=== Sesión terminada ===")
        for k, v in resumen.items():
            logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
