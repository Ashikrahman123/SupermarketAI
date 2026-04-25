from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    category: str
    age_group: str
    confidence: float
    margin: float
    uncertain: bool
    top_label: str
    second_label: str
    threshold: float
    hybrid_alpha: float
    class_probabilities: dict[str, float]


class PredictBusinessResponse(PredictResponse):
    unit_cost: float
    sale_price: float
    quantity: int
    revenue: float
    total_cost: float
    profit: float
    profit_margin_percent: float
    expected_margin_rate: float
    expected_profit: float


class MetadataResponse(BaseModel):
    class_names: list[str]
    image_size: tuple[int, int]
    confidence_threshold: float
    hybrid_alpha: float
    uncertainty_margin: float
