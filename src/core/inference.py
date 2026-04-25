from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from hybrid_inference import build_class_centroids, similarity_probs_for_image
from .config import AppConfig, RuntimeConfig, load_runtime_config


@dataclass(frozen=True)
class PredictionResult:
    category: str
    age_group: str
    confidence: float
    margin: float
    top_idx: int
    second_idx: int
    top_label: str
    second_label: str
    uncertain: bool
    probs: np.ndarray
    class_names: list[str]
    threshold: float
    hybrid_alpha: float


class InferenceEngine:
    def __init__(self, app_config: AppConfig | None = None):
        self.app_config = app_config or AppConfig()
        self.runtime: RuntimeConfig = load_runtime_config(self.app_config)
        self._model = None
        self._centroids = None
        self._centroid_counts: dict[str, int] = {}

    @property
    def centroid_counts(self) -> dict[str, int]:
        return dict(self._centroid_counts)

    def refresh(self) -> None:
        self.runtime = load_runtime_config(self.app_config)
        self._model = None
        self._centroids = None
        self._centroid_counts = {}

    def get_model(self):
        if self._model is not None:
            return self._model

        model_path = next((path for path in self.app_config.model_paths if path.exists()), None)
        if model_path is None:
            raise FileNotFoundError("No model file found (model.keras/model.h5). Run 'make train' first.")

        self._model = tf.keras.models.load_model(model_path, compile=False)
        return self._model

    def get_centroids(self):
        if self._centroids is not None:
            return self._centroids

        centroids, counts = build_class_centroids(self.runtime.class_names, self.runtime.image_size)
        self._centroids = centroids
        self._centroid_counts = counts
        return self._centroids

    @staticmethod
    def predict_with_tta(model, img: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
        rgb = img.convert("RGB")
        variants = [
            rgb,
            ImageOps.mirror(rgb),
            rgb.rotate(8, resample=Image.Resampling.BILINEAR),
            rgb.rotate(-8, resample=Image.Resampling.BILINEAR),
            ImageOps.flip(rgb),
        ]

        batch = []
        for variant in variants:
            resized = variant.resize(image_size)
            batch.append(np.array(resized, dtype=np.float32))

        batch_arr = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.array(batch, dtype=np.float32)
        )
        probs = model.predict(batch_arr, verbose=0)
        return np.mean(probs, axis=0)

    @staticmethod
    def _calibrated_confidence(
        probs: np.ndarray,
        top_idx: int,
        second_idx: int,
    ) -> tuple[float, float, float, float]:
        top_prob = float(probs[top_idx])
        second_prob = float(probs[second_idx])
        class_count = max(1, int(len(probs)))

        if class_count <= 1:
            return top_prob, top_prob, 1.0, top_prob

        # Normalize against multiclass baseline and distribution sharpness.
        uniform = 1.0 / class_count
        lift = (top_prob - uniform) / max(1e-8, 1.0 - uniform)
        lift = float(np.clip(lift, 0.0, 1.0))

        top2_ratio = top_prob / max(1e-8, top_prob + second_prob)

        entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
        max_entropy = float(np.log(class_count))
        entropy_certainty = 1.0 - (entropy / max(1e-8, max_entropy))
        entropy_certainty = float(np.clip(entropy_certainty, 0.0, 1.0))

        raw_calibrated = (
            0.30 * top_prob
            + 0.40 * top2_ratio
            + 0.20 * entropy_certainty
            + 0.10 * lift
        )
        raw_calibrated = float(np.clip(raw_calibrated, 0.0, 1.0))

        # Convert multiclass confidence into a user-facing certainty score.
        boosted = 1.0 / (1.0 + float(np.exp(-16.0 * (raw_calibrated - 0.30))))
        boosted = float(np.clip(boosted, 0.0, 1.0))

        # Keep confident outcomes visually above 80% when class separation is clear.
        if top2_ratio >= 0.56 and top_prob >= (uniform + 0.06):
            boosted = max(boosted, 0.80)

        return boosted, raw_calibrated, top2_ratio, top_prob

    def predict_pil(self, img: Image.Image) -> PredictionResult:
        model = self.get_model()
        model_probs = self.predict_with_tta(model, img, self.runtime.image_size)

        centroids = self.get_centroids()
        sim_probs = similarity_probs_for_image(img, self.runtime.class_names, self.runtime.image_size, centroids)

        if sim_probs is None:
            probs = model_probs
        else:
            probs = self.runtime.hybrid_alpha * model_probs + (1.0 - self.runtime.hybrid_alpha) * sim_probs
            probs = probs / np.sum(probs)

        if len(probs) != len(self.runtime.class_names):
            raise ValueError("Model output classes do not match metadata. Re-train the model with 'make train'.")

        sorted_idx = np.argsort(probs)[::-1]
        top_idx = int(sorted_idx[0])
        second_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top_idx

        confidence, raw_confidence, top2_ratio, top_prob = self._calibrated_confidence(probs, top_idx, second_idx)
        margin = top_prob - float(probs[second_idx])
        top_label = self.runtime.class_names[top_idx]
        second_label = self.runtime.class_names[second_idx]

        # Use a stricter ambiguity check so "low reliability" appears only on genuinely close predictions.
        adaptive_margin = max(0.04, self.app_config.uncertainty_margin * 0.75)
        confidence_gate = max(0.38, self.runtime.confidence_threshold - 0.12)
        uncertain = (
            raw_confidence < confidence_gate
            and margin < adaptive_margin
            and top2_ratio < 0.58
        )

        # Always return the best class; uncertainty is communicated via the reliability flag.
        category = top_label
        age_group = self.runtime.age_map.get(top_label, "All ages")

        return PredictionResult(
            category=category,
            age_group=age_group,
            confidence=confidence,
            margin=margin,
            top_idx=top_idx,
            second_idx=second_idx,
            top_label=top_label,
            second_label=second_label,
            uncertain=uncertain,
            probs=probs,
            class_names=self.runtime.class_names,
            threshold=self.runtime.confidence_threshold,
            hybrid_alpha=self.runtime.hybrid_alpha,
        )

    def predict_path(self, image_path: str | Path) -> PredictionResult:
        with Image.open(image_path) as img:
            return self.predict_pil(img)
