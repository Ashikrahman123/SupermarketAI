import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.config import AppConfig
from src.core.inference import InferenceEngine

MODEL_PATHS = [Path("model.keras"), Path("model.h5")]
ENGINE = InferenceEngine(AppConfig(model_paths=tuple(MODEL_PATHS), metadata_path=Path("model_metadata.json")))


def load_metadata() -> tuple[list[str], tuple[int, int], dict[str, str], float, float]:
    runtime = ENGINE.runtime
    return (
        runtime.class_names,
        runtime.image_size,
        runtime.age_map,
        runtime.confidence_threshold,
        runtime.hybrid_alpha,
    )


def predict_with_tta(model, img_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(img_path) as img:
        return InferenceEngine.predict_with_tta(model, img, image_size)


def predict(img_path: str, threshold: float | None = None) -> None:
    ENGINE.refresh()
    class_names, _, age_map, metadata_threshold, hybrid_alpha = load_metadata()
    if threshold is None:
        threshold = metadata_threshold

    result = ENGINE.predict_path(img_path)

    reliability = "Low" if result.uncertain else "High"
    print(f"Category: {result.top_label}")
    print(f"Age Group: {age_map.get(result.top_label, 'All ages')}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Reliability: {reliability}")

    if result.uncertain:
        print(f"Second likely: {result.second_label} ({float(result.probs[result.second_idx]):.2%})")

    print("Inference: hybrid (trained model + dataset similarity)")

    if result.confidence < threshold:
        print("Note: low confidence prediction. Try a clearer, front-facing product image.")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    predict(image_path)