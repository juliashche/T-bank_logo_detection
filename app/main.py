import io
import os
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import traceback

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MODEL_PATH = os.path.join("weights", "best.pt")

app = FastAPI(title="T-Bank Logo Detection API")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None

# Pydantic модели
class BoundingBox(BaseModel):
    x_min: int = Field(..., ge=0)
    y_min: int = Field(..., ge=0)
    x_max: int = Field(..., ge=0)
    y_max: int = Field(..., ge=0)

class Detection(BaseModel):
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    detections: List[Detection]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str]

# Эндпоинт
@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def detect_logo(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Неподдерживаемый формат файла",
                detail=f"Допустимые форматы: {', '.join(ALLOWED_EXTENSIONS)}"
            ).dict()
        )

    contents = await file.read()
    if not contents:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Пустой файл",
                detail="Файл не содержит данных"
            ).dict()
        )

    # Проверка, что это изображение
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Файл не является изображением",
                detail="Не удалось открыть файл как изображение. Поддерживаются только: JPEG, PNG, BMP, WEBP"
            ).dict()
        )

    if model is None:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Модель не загружена",
                detail=f"Файл весов {MODEL_PATH} не найден или поврежден"
            ).dict()
        )

    try:
        results = model.predict(image, conf=0.5)
    except Exception as e:
        print("Ошибка при детекции YOLO:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Ошибка при обработке изображения моделью",
                detail=str(e)
            ).dict()
        )

    detections = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        x_min, y_min, x_max, y_max = map(int, box[:4])
        detections.append(
            Detection(bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))
        )

    return DetectionResponse(detections=detections)
