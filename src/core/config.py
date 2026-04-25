import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_AGE_MAP = {
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


def discover_dataset_classes(dataset_dir: Path) -> list[str]:
    if not dataset_dir.exists():
        return []
    return sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())


@dataclass(frozen=True)
class AppConfig:
    model_paths: tuple[Path, ...] = (Path("model.keras"), Path("model.h5"))
    metadata_path: Path = Path("model_metadata.json")
    dataset_dir: Path = Path("dataset")
    class_names: tuple[str, ...] = ()
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    age_map: dict[str, str] = None
    confidence_threshold: float = 0.50
    hybrid_alpha: float = 0.35
    uncertainty_margin: float = 0.08

    def __post_init__(self):
        if self.age_map is None:
            object.__setattr__(self, "age_map", DEFAULT_AGE_MAP.copy())


@dataclass(frozen=True)
class RuntimeConfig:
    class_names: list[str]
    image_size: tuple[int, int]
    age_map: dict[str, str]
    confidence_threshold: float
    hybrid_alpha: float


def load_runtime_config(app_config: AppConfig) -> RuntimeConfig:
    discovered_class_names = discover_dataset_classes(app_config.dataset_dir)

    if not app_config.metadata_path.exists():
        fallback_class_names = discovered_class_names or list(app_config.class_names)
        if not fallback_class_names:
            raise FileNotFoundError(
                "No metadata file found and no dataset classes discovered. "
                "Run 'make train' first."
            )

        return RuntimeConfig(
            class_names=fallback_class_names,
            image_size=tuple(app_config.image_size),
            age_map={
                name: app_config.age_map.get(name, "All ages")
                for name in fallback_class_names
            },
            confidence_threshold=app_config.confidence_threshold,
            hybrid_alpha=app_config.hybrid_alpha,
        )

    metadata = json.loads(app_config.metadata_path.read_text(encoding="utf-8"))

    metadata_class_names = metadata.get("class_names")
    if metadata_class_names:
        class_names = list(metadata_class_names)
    elif discovered_class_names:
        class_names = discovered_class_names
    else:
        class_names = list(app_config.class_names)

    if not class_names:
        raise ValueError(
            "Could not determine class names from metadata, dataset, or config."
        )

    metadata_age_map = metadata.get("age_map", {})

    return RuntimeConfig(
        class_names=class_names,
        image_size=tuple(metadata.get("image_size", list(app_config.image_size))),
        age_map={
            name: metadata_age_map.get(name, app_config.age_map.get(name, "All ages"))
            for name in class_names
        },
        confidence_threshold=float(metadata.get("confidence_threshold", app_config.confidence_threshold)),
        hybrid_alpha=float(metadata.get("hybrid_alpha", app_config.hybrid_alpha)),
    )
