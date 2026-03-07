from app.models.patterns.infer_patterns import infer_patterns
from app.models.diseases.infer_diseases import infer_diseases

async def run_patterns_model(image_path: str):
    return infer_patterns(image_path)

async def run_diseases_model(image_path: str):
    return infer_diseases(image_path)