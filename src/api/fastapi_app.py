from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictBusinessResponse,
    PredictResponse,
)
from src.core.business import compute_business_metrics
from src.core.config import AppConfig
from src.core.inference import InferenceEngine

app = FastAPI(title="Supermarket AI API", version="1.0.0")
engine = InferenceEngine(AppConfig())


def _validate_image_content_type(file: UploadFile) -> None:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Only jpeg, png, and webp are supported")


def _build_predict_response(result) -> PredictResponse:
    class_probs = {
        label: float(result.probs[idx])
        for idx, label in enumerate(result.class_names)
    }

    return PredictResponse(
        category=result.category,
        age_group=result.age_group,
        confidence=result.confidence,
        margin=result.margin,
        uncertain=result.uncertain,
        top_label=result.top_label,
        second_label=result.second_label,
        threshold=result.threshold,
        hybrid_alpha=result.hybrid_alpha,
        class_probabilities=class_probs,
    )


async def _predict_from_upload(file: UploadFile):
    _validate_image_content_type(file)
    data = await file.read()
    img = Image.open(BytesIO(data))
    return engine.predict_pil(img)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    runtime = engine.runtime
    return MetadataResponse(
        class_names=runtime.class_names,
        image_size=runtime.image_size,
        confidence_threshold=runtime.confidence_threshold,
        hybrid_alpha=runtime.hybrid_alpha,
        uncertainty_margin=engine.app_config.uncertainty_margin,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    try:
        result = await _predict_from_upload(file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image input: {exc}") from exc

    return _build_predict_response(result)


@app.post("/predict-business", response_model=PredictBusinessResponse)
async def predict_business(
    file: UploadFile = File(...),
    unit_cost: float = Form(...),
    sale_price: float = Form(...),
    quantity: int = Form(...),
) -> PredictBusinessResponse:
    if unit_cost < 0 or sale_price < 0:
        raise HTTPException(status_code=400, detail="unit_cost and sale_price must be >= 0")
    if quantity < 1:
        raise HTTPException(status_code=400, detail="quantity must be >= 1")

    try:
        result = await _predict_from_upload(file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image input: {exc}") from exc

    base_response = _build_predict_response(result)
    business = compute_business_metrics(
        category=result.top_label,
        confidence=result.confidence,
        uncertain=result.uncertain,
        unit_cost=unit_cost,
        sale_price=sale_price,
        quantity=quantity,
    )

    return PredictBusinessResponse(
        **base_response.model_dump(),
        unit_cost=business.unit_cost,
        sale_price=business.sale_price,
        quantity=business.quantity,
        revenue=business.revenue,
        total_cost=business.total_cost,
        profit=business.profit,
        profit_margin_percent=business.profit_margin_percent,
        expected_margin_rate=business.expected_margin_rate,
        expected_profit=business.expected_profit,
    )
