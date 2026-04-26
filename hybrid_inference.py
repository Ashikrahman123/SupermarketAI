from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None  # type: ignore[assignment]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_DIR = Path("dataset")


def _normalized(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return vec
    return vec / norm


def _image_to_batch(img: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    if tf is None:
        raise ModuleNotFoundError("TensorFlow is required for hybrid embeddings.")

    resized = img.convert("RGB").resize(image_size)
    arr = np.array(resized, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


@lru_cache(maxsize=2)
def get_embedding_model(image_size: tuple[int, int]):
    if tf is None:
        raise ModuleNotFoundError("TensorFlow is required for embedding model creation.")

    return tf.keras.applications.MobileNetV2(
        input_shape=image_size + (3,),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )


def build_class_centroids(
    class_names: list[str],
    image_size: tuple[int, int],
    dataset_dir: Path = DATASET_DIR,
) -> tuple[np.ndarray | None, dict[str, int]]:
    if tf is None:
        print("[HybridInference] TensorFlow unavailable; skipping centroid build and using model-only probabilities.")
        return None, {}

    if not dataset_dir.exists():
        print(f"[HybridInference] Dataset directory not found: {dataset_dir}")
        return None, {}

    embedder = get_embedding_model(image_size)
    class_vectors: list[np.ndarray] = []
    class_counts: dict[str, int] = {}

    for class_name in class_names:
        class_dir = dataset_dir / class_name
        vectors = []

        if class_dir.exists():
            for img_path in sorted(class_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in EXTENSIONS:
                    continue
                try:
                    with Image.open(img_path) as img:
                        batch = _image_to_batch(img, image_size)
                    emb = embedder.predict(batch, verbose=0)[0]
                    vectors.append(_normalized(emb))
                except Exception:
                    continue
        else:
            print(f"[HybridInference] Missing class directory for centroid build: {class_dir}")

        class_counts[class_name] = len(vectors)
        if vectors:
            centroid = _normalized(np.mean(np.array(vectors), axis=0))
            class_vectors.append(centroid)
        else:
            print(f"[HybridInference] No valid images found for class '{class_name}'.")
            class_vectors.append(np.zeros((1280,), dtype=np.float32))

    centroids = np.array(class_vectors, dtype=np.float32)
    if not np.any(centroids):
        print("[HybridInference] No class centroids were created; using model-only probabilities.")
        return None, class_counts
    return centroids, class_counts


def similarity_probs_for_image(
    img: Image.Image,
    class_names: list[str],
    image_size: tuple[int, int],
    centroids: np.ndarray | None,
) -> np.ndarray | None:
    if tf is None:
        return None

    if centroids is None:
        return None

    embedder = get_embedding_model(image_size)
    query = embedder.predict(_image_to_batch(img, image_size), verbose=0)[0]
    query = _normalized(query)

    sims = centroids @ query
    sims = np.clip(sims, -1.0, 1.0)

    # Softmax over cosine similarities to produce probability-like scores.
    exp_scores = np.exp((sims - np.max(sims)) / 0.08)
    denom = np.sum(exp_scores)
    if denom <= 1e-12:
        print("[HybridInference] Similarity scores are degenerate; falling back to model-only probabilities.")
        return None
    probs = exp_scores / denom

    if len(probs) != len(class_names):
        return None

    return probs.astype(np.float32)
