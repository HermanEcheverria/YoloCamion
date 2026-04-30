import cv2
import numpy as np
import torch
import torchvision
from typing import Tuple, List
from src.tipos import Clase, Deteccion

# Mapeo de YOLOP (COCO) a clases propias del proyecto
_COCO_A_CLASE: dict[int, Clase] = {
    0: Clase.PEATON,
    2: Clase.VEHICULO,      # car
    3: Clase.MOTOCICLETA,
    5: Clase.VEHICULO,      # bus
    7: Clase.VEHICULO,      # truck
    9: Clase.SEMAFORO,
    11: Clase.SENAL_ALTO,
}

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    Realiza NMS sobre las inferencias de YOLOP.
    prediction: [batch, num_anchors, 85]
    """
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        # Filtro de confianza
        x = x[x[:, 4] > conf_thres]
        if not x.shape[0]:
            continue
            
        # Calcular cajas de [x_center, y_center, w, h] a [x1, y1, x2, y2]
        box = xywh2xyxy(x[:, :4])
        
        # Encontrar la mejor clase
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        if not x.shape[0]:
            continue
            
        # Batched NMS
        c = x[:, 5:6] * 4096  # offset por clase
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i]
        
    return output

class InferenciaYOLOP:
    """Carga YOLOP (Panoptic Driving Perception) y procesa objetos, area manejable y lineas."""

    def __init__(self, conf_min: float = 0.35, imgsz: int = 640, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_min = conf_min
        self.imgsz = imgsz
        self.modelo = None

    def cargar(self):
        """Carga el modelo YOLOP preentrenado usando PyTorch Hub."""
        print("[YOLOP] Cargando modelo Panóptico (puede tardar en la primera ejecución)...")
        # Silenciamos warnings sobre untrusted repos
        import warnings
        warnings.filterwarnings("ignore", message="You are about to download and run code from an untrusted repository")
        
        # Intentamos cargar de Torch Hub
        self.modelo = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True, trust_repo=True)
        self.modelo.to(self.device)
        self.modelo.eval()
        print("[YOLOP] Modelo cargado exitosamente en", self.device)

    def preprocesar(self, imagen: np.ndarray) -> Tuple[torch.Tensor, Tuple[float, float], Tuple[int, int]]:
        """Prepara la imagen para el modelo y guarda escalas para postprocesar."""
        shape_original = imagen.shape[:2]
        
        img = cv2.resize(imagen, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # HWC a CHW
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        
        img_tensor = torch.from_numpy(img).float().to(self.device)
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
        # Normalizacion de ImageNet requerida por YOLOP
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        escala_x = shape_original[1] / self.imgsz
        escala_y = shape_original[0] / self.imgsz
        
        return img_tensor, (escala_x, escala_y), shape_original

    def procesar_frame(self, imagen: np.ndarray) -> Tuple[list[Deteccion], np.ndarray, np.ndarray]:
        """
        Infiere todo en una sola pasada.
        Retorna:
            - detecciones: Lista de objetos (Deteccion)
            - area_manejable: Mascara binaria (shape_original)
            - lineas_carril: Mascara binaria (shape_original)
        """
        if self.modelo is None:
            raise RuntimeError("Llama a cargar() antes de procesar_frame()")
            
        img_tensor, escalas, shape_orig = self.preprocesar(imagen)
        
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = self.modelo(img_tensor)

        # 1. Postprocesar Detecciones (Bounding Boxes)
        inf_out, _ = det_out  # YOLOP retorna (inference_out, training_out)
        pred = non_max_suppression(inf_out, conf_thres=self.conf_min, iou_thres=0.45)
        
        detecciones: list[Deteccion] = []
        if len(pred) > 0 and pred[0] is not None:
            # Rescalar cajas al tamaño original de la imagen
            escala_x, escala_y = escalas
            for *xyxy, conf, cls in pred[0]:
                id_coco = int(cls)
                clase_proyect = _COCO_A_CLASE.get(id_coco, Clase.DESCONOCIDO)
                
                x1 = int(xyxy[0] * escala_x)
                y1 = int(xyxy[1] * escala_y)
                x2 = int(xyxy[2] * escala_x)
                y2 = int(xyxy[3] * escala_y)
                
                # Clip
                x1 = max(0, min(x1, shape_orig[1]))
                y1 = max(0, min(y1, shape_orig[0]))
                x2 = max(0, min(x2, shape_orig[1]))
                y2 = max(0, min(y2, shape_orig[0]))
                
                area = (x2 - x1) * (y2 - y1)
                
                if area > 0:
                    detecciones.append(
                        Deteccion(
                            clase=clase_proyect,
                            caja=(x1, y1, x2, y2),
                            confianza=float(conf),
                            area=area,
                        )
                    )

        # 2. Postprocesar Area Manejable (Drivable Area)
        da_predict = da_seg_out[:, 1, :, :] > da_seg_out[:, 0, :, :]
        da_seg_mask = da_predict.byte().cpu().numpy()[0]
        # Rescalar a original
        da_seg_mask = cv2.resize(da_seg_mask, (shape_orig[1], shape_orig[0]), interpolation=cv2.INTER_NEAREST)

        # 3. Postprocesar Lineas de Carril
        ll_predict = ll_seg_out[:, 1, :, :] > ll_seg_out[:, 0, :, :]
        ll_seg_mask = ll_predict.byte().cpu().numpy()[0]
        ll_seg_mask = cv2.resize(ll_seg_mask, (shape_orig[1], shape_orig[0]), interpolation=cv2.INTER_NEAREST)

        # Cerrar brechas en las líneas de carril: YOLOP pierde píxeles intermedios
        # en curvas porque la línea se vuelve oblicua. Un cierre morfológico con
        # kernel elíptico vertical conecta los segmentos y permite que el ajuste
        # polinomial tenga suficientes puntos para predecir la trayectoria.
        _kernel_curva = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 19))
        ll_seg_mask = cv2.morphologyEx(ll_seg_mask, cv2.MORPH_CLOSE, _kernel_curva)

        return detecciones, da_seg_mask, ll_seg_mask
