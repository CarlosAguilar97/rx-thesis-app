import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "best_multilabelv2.pth"
OUTPUTS_DIR = Path("outputs_infer")


# -------------------------
# Cargar modelo
# -------------------------
def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    labels = ckpt["labels"]
    img_size = int(ckpt["img_size"])

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(labels))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model, labels, img_size


MODEL, LABELS, IMG_SIZE = load_model(str(CKPT_PATH))


# -------------------------
# Preprocesado
# -------------------------
def preprocess(image_path: str, img_size: int):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"No pude leer: {image_path}")

    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    x = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.tensor(x).unsqueeze(0)

    return rgb, x


# -------------------------
# Reporte radiomorfológico
# -------------------------
def build_report(probs: dict, thr_global: float = 0.5):
    def p(k):
        return float(probs.get(k, 0.0))

    def on(k):
        return p(k) >= float(thr_global)

    dist_parts = []
    if on("upper_lung_predominance"):
        dist_parts.append("predominio en campos superiores")
    if on("lower_lung_predominance"):
        dist_parts.append("predominio basal")
    if on("perihilar_distribution"):
        dist_parts.append("distribución perihiliar")
    if on("peripheral_distribution"):
        dist_parts.append("distribución periférica/subpleural")
    if on("diffuse_distribution"):
        dist_parts.append("distribución difusa")
    if on("asymmetric_distribution"):
        dist_parts.append("distribución asimétrica")

    dist_txt = (" (" + ", ".join(dist_parts) + ").") if dist_parts else ""

    findings = []

    if on("interstitial_pattern") or on("diffuse_distribution"):
        findings.append("Patrón intersticial." + dist_txt)

    if on("reticular_pattern"):
        findings.append("Patrón reticular." + dist_txt)

    if on("nodular_density") or on("nodular_pattern"):
        findings.append("Densidades nodulares." + dist_txt)

    if on("alveolar_pattern"):
        findings.append("Opacidades de tipo alveolar." + dist_txt)

    pleura_bits = []
    if on("pleural_effusion"):
        pleura_bits.append("derrame pleural")
    if on("pleural_thickening"):
        pleura_bits.append("engrosamiento pleural")
    if pleura_bits:
        findings.append("Pleura: " + "; ".join(pleura_bits) + ".")

    vol_bits = []
    if on("volume_loss"):
        vol_bits.append("disminución de volumen pulmonar")
    if on("mediastinal_shift"):
        vol_bits.append("desplazamiento mediastínico")
    if vol_bits:
        findings.append("Cambios de volumen/posición: " + "; ".join(vol_bits) + ".")

    if on("cardiomegaly"):
        findings.append("Cardiomediastino: cardiomegalia.")

    if not findings:
        findings.append("Sin hallazgos patológicos evidentes en esta evaluación automática.")

    priority = [
        "pleural_effusion",
        "alveolar_pattern",
        "nodular_density",
        "interstitial_pattern",
        "hyperlucency",
        "volume_loss",
        "reticular_pattern",
        "pleural_thickening",
        "mediastinal_shift",
        "cardiomegaly",
    ]

    impression = []
    for k in priority:
        if k in probs and on(k):
            impression.append(k)
        if len(impression) >= 3:
            break

    if not impression:
        top1 = max(probs.items(), key=lambda kv: kv[1])[0]
        impression = [top1]

    key_to_phrase = {
        "pleural_effusion": "Derrame pleural.",
        "alveolar_pattern": "Opacidades de tipo alveolar.",
        "nodular_density": "Densidades nodulares.",
        "interstitial_pattern": "Patrón intersticial.",
        "hyperlucency": "Hiperclaridad/hiperlucencia (patrón).",
        "volume_loss": "Disminución de volumen pulmonar.",
        "reticular_pattern": "Patrón reticular.",
        "pleural_thickening": "Engrosamiento pleural.",
        "mediastinal_shift": "Desplazamiento mediastínico.",
        "cardiomegaly": "Cardiomegalia.",
    }

    impression_lines = [key_to_phrase.get(k, k) for k in impression[:3]]

    return {
        "ESTUDIO": "Radiografía de tórax",
        "HALLAZGOS": findings,
        "IMPRESION": impression_lines,
        "DISCLAIMER": "Salida automática basada en patrones radiomorfológicos; correlacionar con clínica."
    }


# -------------------------
# Grad-CAM
# -------------------------
class GradCAMAutograd:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._hook = target_layer.register_forward_hook(self._save_acts)

    def _save_acts(self, module, inp, out):
        self.activations = out
        self.activations.retain_grad()

    def close(self):
        self._hook.remove()

    def generate(self, x: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        score = logits[0, class_index]

        grads = torch.autograd.grad(
            outputs=score,
            inputs=self.activations,
            retain_graph=True,
            create_graph=False
        )[0]

        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)[0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    cam_resized = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    out = (rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------
# Inferencia principal
# -------------------------
def infer_patterns(image_path: str):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    rgb, x = preprocess(image_path, IMG_SIZE)
    x = x.to(DEVICE)
    x.requires_grad_(True)

    with torch.no_grad():
        logits = MODEL(x)[0]
        probs = torch.sigmoid(logits).cpu().numpy()

    probs_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    topk = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)[:8]

    report = build_report(probs_dict, thr_global=0.5)

    # Grad-CAM sobre top 3
    target_layer = MODEL.features.denseblock4
    cam_engine = GradCAMAutograd(MODEL, target_layer)

    _ = MODEL(x)  # forward para capturar activaciones

    cam_paths = {}
    unique_prefix = uuid.uuid4().hex[:8]

    for lab, _prob in topk[:3]:
        idx = LABELS.index(lab)
        cam = cam_engine.generate(x, class_index=idx)
        overlay = overlay_cam(rgb, cam, alpha=0.35)

        filename = f"{unique_prefix}_gradcam_{lab}.png"
        out_path = OUTPUTS_DIR / filename

        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cam_paths[lab] = f"/outputs/{filename}"

    cam_engine.close()

    return {
        "pred1_label": topk[0][0],
        "pred1_prob": float(topk[0][1]),
        "pred2_label": topk[1][0] if len(topk) > 1 else None,
        "pred2_prob": float(topk[1][1]) if len(topk) > 1 else None,
        "probabilities": probs_dict,
        "report": report,
        "gradcam_images": cam_paths,
    }