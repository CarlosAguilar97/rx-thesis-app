from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os
from app.config import UPLOAD_DIR, OUTPUTS_DIR

from app.db import (
    init_db,
    create_case,
    save_patterns_result,
    save_diseases_result,
    get_history,
)
from app.models_logic import run_patterns_model, run_diseases_model
from app.schemas import AnalyzeResponse

app = FastAPI(title="RX Tórax API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.on_event("startup")
async def startup_event():
    await init_db()


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    original_description: str = Form("")
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo inválido")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Solo se permiten PNG/JPG")

    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = Path(UPLOAD_DIR) / unique_name

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    id_case = await create_case(
        image_name=file.filename,
        image_path=str(file_path),
        original_description=original_description
    )

    patterns_result = await run_patterns_model(str(file_path))
    diseases_result = await run_diseases_model(str(file_path))

    await save_patterns_result(
        id_case=id_case,
        model_version="best_multilabelv2",
        pred1_label=patterns_result["pred1_label"],
        pred1_prob=patterns_result["pred1_prob"],
        pred2_label=patterns_result["pred2_label"],
        pred2_prob=patterns_result["pred2_prob"],
        probs_json=patterns_result["probabilities"],
        gradcam_json=patterns_result.get("gradcam_images"),
        report_json=patterns_result.get("report"),
    )

    await save_diseases_result(
        id_case=id_case,
        model_version="clinical_model",
        pred1_label=diseases_result["pred1_label"],
        pred1_prob=diseases_result["pred1_prob"],
        pred2_label=diseases_result["pred2_label"],
        pred2_prob=diseases_result["pred2_prob"],
        probs_json=diseases_result["probabilities"],
    )

    return AnalyzeResponse(
        id_case=id_case,
        image_name=file.filename,
        image_path=f"/uploads/{unique_name}",
        patterns=patterns_result,
        diseases=diseases_result,
    )


@app.get("/history")
async def history(limit: int = 20):
    rows = await get_history(limit)
    data = []
    for row in rows:
        data.append({
            "id_case": row[0],
            "image_name": row[1],
            "image_path": row[2],
            "created_at": row[3],
            "patterns_pred1": row[4],
            "patterns_prob1": row[5],
            "diseases_pred1": row[6],
            "diseases_prob1": row[7],
        })
    return data