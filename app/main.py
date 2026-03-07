from pathlib import Path
import shutil
import uuid
import traceback
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RX Tórax API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.on_event("startup")
async def startup_event():
    await init_db()
    logger.info("Startup completado correctamente")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    original_description: str = Form(...)
):
    try:
        logger.info("Entró a /analyze")
        logger.info(f"Archivo recibido: {file.filename}")
        logger.info(f"Descripción recibida: {original_description!r}")

        if not file.filename:
            raise HTTPException(status_code=400, detail="Archivo inválido")

        if not original_description or not original_description.strip():
            raise HTTPException(status_code=400, detail="La descripción es obligatoria")

        ext = Path(file.filename).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg"]:
            raise HTTPException(status_code=400, detail="Solo se permiten PNG/JPG")

        unique_name = f"{uuid.uuid4().hex}{ext}"
        file_path = Path(UPLOAD_DIR) / unique_name

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Imagen guardada en: {file_path}")

        id_case = await create_case(
            image_name=file.filename,
            image_path=str(file_path),
            original_description=original_description
        )

        logger.info(f"Caso creado con id_case={id_case}")

        patterns_result = await run_patterns_model(str(file_path))
        logger.info("Modelo patterns ejecutado correctamente")

        diseases_result = await run_diseases_model(str(file_path))
        logger.info("Modelo diseases ejecutado correctamente")

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
        logger.info("Resultados patterns guardados")

        await save_diseases_result(
            id_case=id_case,
            model_version="clinical_model",
            pred1_label=diseases_result["pred1_label"],
            pred1_prob=diseases_result["pred1_prob"],
            pred2_label=diseases_result["pred2_label"],
            pred2_prob=diseases_result["pred2_prob"],
            probs_json=diseases_result["probabilities"],
        )
        logger.info("Resultados diseases guardados")

        return AnalyzeResponse(
            id_case=id_case,
            image_name=file.filename,
            image_path=f"/uploads/{unique_name}",
            patterns=patterns_result,
            diseases=diseases_result,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error en /analyze")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno en analyze: {str(e)}")


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