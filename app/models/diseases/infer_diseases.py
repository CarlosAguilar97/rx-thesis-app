from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

IMG_SIZE = 512
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).resolve().parent / "clinical_model.pth"


def build_model():
    model = models.densenet121(weights=None)

    model.features.conv0 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    model.classifier = nn.Linear(model.classifier.in_features, len(DISEASES))
    return model


MODEL = build_model()
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
MODEL = MODEL.to(DEVICE)
MODEL.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.25])
])


def infer_diseases(image_path: str):
    img = Image.open(image_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(img_tensor)
        pred = torch.sigmoid(logits).cpu().numpy()[0]

    probabilities = {
        DISEASES[i]: float(pred[i]) for i in range(len(DISEASES))
    }

    ordered = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    topk = ordered[:2]

    positives = [
        {"label": label, "prob": float(prob)}
        for label, prob in ordered
        if prob >= THRESHOLD
    ]

    return {
        "pred1_label": topk[0][0],
        "pred1_prob": float(topk[0][1]),
        "pred2_label": topk[1][0] if len(topk) > 1 else None,
        "pred2_prob": float(topk[1][1]) if len(topk) > 1 else None,
        "probabilities": probabilities,
        "positives": positives,
        "report": None,
        "gradcam_images": None,
    }