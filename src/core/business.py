from dataclasses import dataclass


# Simple category-wise expected margin priors for demo/business simulation.
CATEGORY_MARGIN_PRIORS = {
    "baby products": 0.18,
    "bakery": 0.22,
    "beauty": 0.34,
    "electronics": 0.15,
    "grocery": 0.11,
    "household": 0.19,
    "snacks": 0.24,
    "stationaries": 0.27,
    "toys": 0.29,
}


@dataclass(frozen=True)
class BusinessMetrics:
    unit_cost: float
    sale_price: float
    quantity: int
    revenue: float
    total_cost: float
    profit: float
    profit_margin_percent: float
    expected_margin_rate: float
    expected_profit: float


def _resolve_prior_margin(category: str) -> float:
    return CATEGORY_MARGIN_PRIORS.get(category.strip().lower(), 0.18)


def expected_margin_rate(category: str, confidence: float, uncertain: bool) -> float:
    base_rate = _resolve_prior_margin(category)

    # Confidence adjusts expected margin mildly; uncertainty penalizes it.
    confidence_adjustment = (confidence - 0.5) * 0.08
    uncertainty_penalty = 0.06 if uncertain else 0.0

    adjusted = base_rate + confidence_adjustment - uncertainty_penalty
    return max(0.03, min(0.55, adjusted))


def compute_business_metrics(
    category: str,
    confidence: float,
    uncertain: bool,
    unit_cost: float,
    sale_price: float,
    quantity: int,
) -> BusinessMetrics:
    quantity = max(1, int(quantity))
    unit_cost = max(0.0, float(unit_cost))
    sale_price = max(0.0, float(sale_price))

    revenue = sale_price * quantity
    total_cost = unit_cost * quantity
    profit = revenue - total_cost
    profit_margin_percent = (profit / revenue * 100.0) if revenue > 0 else 0.0

    expected_rate = expected_margin_rate(category, confidence, uncertain)
    expected_profit = revenue * expected_rate

    return BusinessMetrics(
        unit_cost=unit_cost,
        sale_price=sale_price,
        quantity=quantity,
        revenue=revenue,
        total_cost=total_cost,
        profit=profit,
        profit_margin_percent=profit_margin_percent,
        expected_margin_rate=expected_rate,
        expected_profit=expected_profit,
    )
