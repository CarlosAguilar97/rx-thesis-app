from pydantic import BaseModel
from typing import Optional, Dict, List


class PositiveLabel(BaseModel):
    label: str
    prob: float


class PredictionResult(BaseModel):
    pred1_label: str
    pred1_prob: float
    pred2_label: Optional[str] = None
    pred2_prob: Optional[float] = None
    probabilities: Dict[str, float]
    positives: Optional[List[PositiveLabel]] = None
    report: Optional[dict] = None
    gradcam_images: Optional[Dict[str, str]] = None


class AnalyzeResponse(BaseModel):
    id_case: int
    image_name: str
    image_path: str
    patterns: PredictionResult
    diseases: PredictionResult