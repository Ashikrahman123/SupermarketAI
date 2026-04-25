import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

DATASET_DIR = Path("dataset")
MODEL_PATH = Path("model.keras")
METADATA_PATH = Path("model_metadata.json")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
VAL_SPLIT = 0.25
CONFIDENCE_THRESHOLD = 0.50
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

AGE_MAP = {
    "Baby products": "0-5 years",
    "bakery": "6+ years",
    "Beauty": "13+ years",
    "electronics": "12+ years",
    "Grocery": "All ages",
    "household": "18+ years",
    "Snacks": "6+ years",
    "Stationaries": "5-25 years",
    "Toys": "3-12 years",
}


def discover_classes_and_counts(dataset_dir: Path) -> tuple[list[str], dict[str, int]]:
    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found under: {dataset_dir}")

    class_names = [path.name for path in class_dirs]
    counts: dict[str, int] = {}
    empty_classes: list[str] = []
    invalid_images: list[Path] = []

    for class_dir in class_dirs:
        valid_count = 0
        for img_path in class_dir.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in EXTENSIONS:
                continue

            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_count += 1
            except Exception:
                invalid_images.append(img_path)

        counts[class_dir.name] = valid_count
        if valid_count == 0:
            empty_classes.append(class_dir.name)

    if empty_classes:
        raise ValueError(
            "These class folders have no valid images: " + ", ".join(empty_classes)
        )

    if invalid_images:
        preview = ", ".join(str(path) for path in invalid_images[:5])
        print(
            f"Warning: skipped {len(invalid_images)} invalid image(s) during precheck. "
            f"Examples: {preview}"
        )

    return class_names, counts


def count_labels(train_ds: tf.data.Dataset, num_classes: int) -> list[int]:
    counts = [0] * num_classes
    for _, labels in train_ds.unbatch():
        counts[int(labels.numpy())] += 1
    return counts


def compute_class_weight(class_counts: list[int]) -> dict[int, float]:
    num_classes = len(class_counts)
    total = sum(class_counts)
    return {
        idx: (total / (num_classes * max(count, 1)))
        for idx, count in enumerate(class_counts)
    }


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    discovered_class_names, discovered_counts = discover_classes_and_counts(DATASET_DIR)
    print(f"Discovered classes ({len(discovered_class_names)}): {discovered_class_names}")
    print(f"Detected image counts per class: {discovered_counts}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        class_names=discovered_class_names,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        class_names=discovered_class_names,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names
    if class_names != discovered_class_names:
        raise ValueError("Discovered classes do not match training dataset class order.")

    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Need at least 2 classes in dataset/ to train a classifier.")

    train_ds = train_ds.ignore_errors()
    val_ds = val_ds.ignore_errors()

    class_counts = count_labels(train_ds, num_classes)
    class_weight = compute_class_weight(class_counts)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(512, seed=SEED, reshuffle_each_iteration=True).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.04, 0.04),
            layers.RandomContrast(0.12),
        ]
    )

    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"Training with classes: {class_names}")
    print(f"Class counts (training split): {dict(zip(class_names, class_counts))}")
    print(f"Class weights: {class_weight}")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        callbacks=callbacks_phase1,
        class_weight=class_weight,
        verbose=1,
    )

    # Fine-tune top layers of MobileNetV2 for better adaptation on the small custom dataset.
    base_model.trainable = True
    for layer in base_model.layers[:-60]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-6),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=28,
        callbacks=callbacks_phase2,
        class_weight=class_weight,
        verbose=1,
    )

    model.save(MODEL_PATH)

    resolved_age_map = {name: AGE_MAP.get(name, "All ages") for name in class_names}

    metadata = {
        "class_names": class_names,
        "image_size": list(IMAGE_SIZE),
        "age_map": resolved_age_map,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "hybrid_alpha": 0.35,
        "class_counts_train": dict(zip(class_names, class_counts)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model saved to {MODEL_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")
    print(f"Resolved age mapping: {resolved_age_map}")


if __name__ == "__main__":
    main()