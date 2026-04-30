# YoloCamion — Conducción autónoma ETS2 con YOLOP + PurePursuit

## Qué hace este proyecto

Piloto autónomo para **Euro Truck Simulator 2** que:
1. Captura la pantalla del juego en tiempo real
2. Usa **YOLOP** (YOLO-based Panoptic Driving Perception) para detectar en UNA sola pasada:
   - Líneas de carril (`ll_mask`) → ajuste polinomial para curvas
   - Área manejable (`da_mask`) → fallback cuando faltan líneas
   - Vehículos/peatones/semáforos → decisión FSM
3. Calcula el giro del volante con **PurePursuitVisual** (3 niveles de señal)
4. Envía comandos a un **gamepad Xbox virtual** (vgamepad) que el juego interpreta

## Cómo correr el piloto hoy

```bash
# 1. Abre ETS2 primero, coloca el camión en una carretera
# 2. En terminal, desde la raíz del repo:
python scripts/ejecutar_piloto.py --fuente ventana --control gamepad --delay 5 --debug-carril

# Con delay 5s para que puedas cambiar al juego
# --debug-carril muestra curvatura y desviación cada 30 frames
```

**Parar de emergencia:** presiona `F12` en cualquier momento.

## Cómo diagnosticar curvas (nuevo)

```bash
# Analiza un video grabado y genera imágenes de debug con polinomios anotados:
python scripts/probar_curvas.py --video datos/videos/ets2_volvo_fh16.f140.m4a

# Ver en tiempo real (requiere display X11):
python scripts/probar_curvas.py --video mi_video.mp4 --mostrar --cada 1

# Las imágenes se guardan en datos/evidencia/curvas/
```

**Qué buscar en las imágenes de debug:**
- Líneas azul/naranja (polinomios) deben seguir los bordes del carril
- Punto cian = look-ahead point → debe estar dentro del carril
- `fuente=poly` = ajuste polinomial activo (mejor señal en curvas)
- `fuente=da_mask` = fallback (aceptable)
- `fuente=memoria` = señal perdida (revisar si supera 10%)

## Cómo correr los tests

```bash
pytest tests/ -v
# Todos deben pasar. Si falla test_pure_pursuit → revisar cambios en pure_pursuit.py
```

---

## Arquitectura del pipeline

```
Pantalla ETS2
    ↓ (FuenteVentana)
Frame BGR 1920×1080
    ↓ (InferenciaYOLOP — cada 2 frames)
    ├─ da_mask (área manejable, uint8)
    ├─ ll_mask (líneas de carril, uint8)  ← cierre morfológico en curvas
    └─ detecciones (bbox de vehículos/peatones)
    ↓ (PurePursuitVisual)
    ├─ Nivel 1a: ajuste polinomial ll_mask → centro exacto en curvas
    ├─ Nivel 1b: filas de ll_mask → fallback con pocos píxeles
    ├─ Nivel 2: centroide da_mask → fallback sin líneas
    └─ Nivel 3: memoria con decaimiento → carril perdido
    ↓ (EMA dinámica — alpha sube en curvas)
desv_volante ∈ [-1, 1]
    ↓ (FSM + override de carril)
SetpointControl
    ↓ (ControladorGamepadPID — 3 PIDs)
vgamepad Xbox → ETS2
```

## Qué mejoró en esta versión (2026-04-30)

### Problema anterior
El camión se salía en curvas porque:
- La señal de ll_mask solo muestreaba 5 filas en el look-ahead (~72-85% de imagen)
- En una curva pronunciada, esas filas ya están fuera del carril visible → señal perdida
- El EMA fijo de 0.12 era demasiado conservador para reaccionar rápido en curva

### Soluciones implementadas

| Cambio | Archivo | Efecto |
|--------|---------|--------|
| **Ajuste polinomial** de ll_mask (Nivel 1a) | `src/control/pure_pursuit.py` | Usa TODOS los píxeles visibles para predecir el centro en curvas — no depende solo de las filas del look-ahead |
| **Cierre morfológico** en ll_mask | `src/percepcion/yolop_inference.py` | Conecta segmentos discontinuos de línea (YOLOP pierde píxeles en curvas oblicuas) |
| **EMA dinámica** según curvatura | `scripts/ejecutar_piloto.py` | Alpha 0.12→0.35 cuando la curva es cerrada — reacción más rápida |
| **Zona muerta adaptativa** | `scripts/ejecutar_piloto.py` | ±0.05 en recta → ±0.02 en curva — correcciones finas antes de salirse |
| **Script `probar_curvas.py`** | `scripts/probar_curvas.py` | Herramienta visual para diagnosticar comportamiento en curvas |

---

## Calibración rápida si el camión oscila en recta

El oscilador principal es el **Kp del volante** en `src/control/gamepad_pid.py`:
```python
_CFG_VOLANTE_DEFAULT = ConfigPID(kp=0.50, ki=0.010, kd=0.10)
```
- Si oscila → bajar `kp` a 0.40
- Si responde lento → subir `kp` a 0.60 (pero riesgo de oscilación)
- El `kd` amortigua la oscilación — subir si hay rebotes

## Calibración para curvas cerradas

Si sigue saliéndose en curvas cerradas, en `src/control/pure_pursuit.py`:
```python
_FILA_CERCA = 0.85  # subir a 0.88 para mirar más cerca (más reacción anticipada)
_MIN_LL_POLY = 30   # bajar a 20 si hay pocas líneas detectadas
```

## Estructura de archivos

```
src/
  control/
    pure_pursuit.py    ← lógica de lane-following
    gamepad_pid.py     ← controlador PID + vgamepad
  percepcion/
    yolop_inference.py ← YOLOP: detección + máscaras
    detector.py        ← YOLOv8 para objetos (ultralytics)
  decision/
    fsm.py             ← 12 reglas de decisión + TTC
scripts/
  ejecutar_piloto.py   ← punto de entrada principal
  probar_curvas.py     ← diagnóstico visual de curvas (nuevo)
  debug_roi_carriles.py
  benchmark_fps.py
config/
  default.yaml         ← parámetros por defecto
tests/                 ← pytest, correr con: pytest tests/ -v
```
