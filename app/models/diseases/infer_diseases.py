import base64
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms, models
from PIL import Image

# --- CONFIGURACIÓN DE DISPOSITIVO HÍBRIDO (LOCAL vs NUBE) ---
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print("🚀 Hardware detected: Usando GPU local mediante DirectML")
except (ImportError, RuntimeError):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"☁️ Environment detected: Usando {DEVICE}")

# --- CONFIGURACIÓN ---
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

DESC_CLINICA = {
    'Effusion': "opacidad homogénea compatible con presencia de líquido en el espacio pleural",
    'Infiltration': "presencia de opacidades de aspecto alveolar-intersticial",
    'Cardiomegaly': "incremento global de la silueta cardíaca (índice cardiotorácico aumentado)",
    'Atelectasis': "pérdida de volumen segmentario con signos de colapso pulmonar",
    'Pneumothorax': "presencia de aire en espacio pleural con retracción del parénquima",
    'Pneumonia': "consolidación del parénquima pulmonar con presencia de broncograma aéreo",
    'Edema': "signos de redistribución vascular y cefalización de la trama",
    'Consolidation': "imagen de densidad aumentada con borramiento de estructuras vasculares",
    'Mass': "lesión de aspecto expansivo de gran tamaño e irregular",
    'Nodule': "imagen radiopaca focal de bordes definidos"
}

IMG_SIZE = 512
MODEL_PATH = Path(__file__).resolve().parent / "modelo_chest_ray_limpio.pth"

# --- FUNCIONES DE APOYO ---

def to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def build_model():
    model = models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, len(DISEASES))
    
    # Carga segura: Siempre a CPU primero para evitar conflictos
    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
    else:
        print(f"⚠️ Alerta: No se encontró el archivo de pesos en {MODEL_PATH}")
        
    return model.to(DEVICE).eval()

MODEL = build_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# --- GENERADOR DE INFORME DINÁMICO ---
def generar_reporte_dinamico(top_results, heatmap):
    mitad = heatmap.shape[1] // 2
    lado = " IZQUIERDO" if np.sum(heatmap[:, :mitad]) > np.sum(heatmap[:, mitad:]) else " DERECHO"
    
    cuerpo = "EL ESTUDIO RADIOLÓGICO DEL TÓRAX MUESTRA:\n\n"
    for i, (name, prob) in enumerate(top_results):
        adj = "Extensa" if prob > 0.40 else "Moderada" if prob > 0.20 else "Discreta"
        desc = DESC_CLINICA.get(name, f"signos de {name.lower()}")
        ubicacion = lado if i == 0 else "" 
        cuerpo += f"* {adj} {desc}{ubicacion}.\n"

    cuerpo += "\nResto de campos pulmonares sin alteraciones evidentes."
    return cuerpo

# --- FUNCIÓN DE INFERENCIA PRINCIPAL ---
def infer_diseases(image_path: str):
    raw_img = Image.open(image_path).convert("L")
    w_orig, h_orig = raw_img.size
    img_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

    # Hook para Grad-CAM manual
    activations = []
    def hook_fn(m, i, o): activations.append(o.detach())
    handle = MODEL.features.norm5.register_forward_hook(hook_fn)

    with torch.no_grad():
        logits = MODEL(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    handle.remove()

    # 1. Procesar probabilidades
    probabilities = {DISEASES[i]: float(probs[i]) for i in range(len(DISEASES))}
    ordered = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    topk = ordered[:3] 

    # 2. Definir variable 'positives' (LO QUE FALTABA)
    positives = [
        {"label": label, "prob": float(prob)}
        for label, prob in ordered
        if prob >= 0.20
    ]

    # 3. Generar Heatmap (Grad-CAM)
    target_idx = DISEASES.index(topk[0][0])
    weights = MODEL.classifier.weight[target_idx, :].detach()
    act = activations[0].squeeze(0)
    
    cam = torch.zeros(act.shape[1:], device=DEVICE)
    for i, w_val in enumerate(weights):
        cam += w_val * act[i, :, :]
    
    cam = torch.clamp(cam, min=0).cpu().numpy()
    cam = (cam - np.min(cam)) / (np.max(cam) + 1e-10)

    # Convertir Heatmap a Base64
    heatmap_res = cv2.resize(cam, (w_orig, h_orig))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor(np.array(raw_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    gradcam_b64 = to_base64(overlay_pil)

    # 4. Generar el reporte
    reporte_texto = generar_reporte_dinamico(topk, cam)

    # RETORNO FINAL ADAPTADO AL SCHEMA
    return {
        "pred1_label": topk[0][0],
        "pred1_prob": float(topk[0][1]),
        "pred2_label": topk[1][0] if len(topk) > 1 else None,
        "pred2_prob": float(topk[1][1]) if len(topk) > 1 else None,
        "probabilities": probabilities,
        "positives": positives,
        # Devolvemos diccionarios vacíos para cumplir con el contrato de Pydantic
        # y no generar errores de validación en Render
        "report": {}, 
        "gradcam_images": {}
    }