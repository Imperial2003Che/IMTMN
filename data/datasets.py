"""Datasets for cross-view image matching."""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UAV_TOKENS = {
    "uav", "drone", "drone_view", "drone_view_512", "query_drone", "gallery_drone",
    "4k_drone", "150m", "200m", "250m", "300m",
}
SAT_TOKENS = {
    "sat", "satellite", "satellite_view", "gallery_satellite", "query_satellite", "map",
}
GROUND_TOKENS = {"ground", "street", "google", "query_street", "gallery_street", "panorama"}
TRAIN_TOKENS = {"train", "training"}
VAL_TOKENS = {"val", "valid", "validation"}
TEST_TOKENS = {"test", "testing", "query", "gallery"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    dataset: str
    label: str
    view: str
    explicit_split: str | None = None


def _stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _normalise_token(token: str) -> str:
    return token.lower().replace("-", "_").replace(" ", "_")


def _view_from_parts(parts: Iterable[str]) -> tuple[str | None, int | None]:
    normalised = [_normalise_token(part) for part in parts]
    for idx, part in enumerate(normalised):
        if part in UAV_TOKENS:
            return "uav", idx
        if part in SAT_TOKENS:
            return "sat", idx
        if part in GROUND_TOKENS:
            return "ground", idx
        if "drone" in part or "uav" in part:
            return "uav", idx
        if "satellite" in part or part.startswith("sat_") or part.endswith("_sat"):
            return "sat", idx
        if "street" in part or "ground" in part or "google" in part:
            return "ground", idx
    return None, None


def _explicit_split(parts: Iterable[str]) -> str | None:
    for raw in parts:
        part = _normalise_token(raw)
        if part in TRAIN_TOKENS:
            return "train"
        if part in VAL_TOKENS:
            return "val"
        if part in TEST_TOKENS or part.startswith("query_") or part.startswith("gallery_"):
            return "test"
    return None


def _label_from_parts(parts: list[str], view_idx: int | None, path: Path) -> str:
    if view_idx is not None and view_idx + 1 < len(parts) - 1:
        candidate = parts[view_idx + 1]
        if _normalise_token(candidate) not in TRAIN_TOKENS | VAL_TOKENS | TEST_TOKENS:
            return candidate
    return path.parent.name


def _read_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _discover_records(root_dir: str | os.PathLike, dataset_name: str) -> list[ImageRecord]:
    root = Path(root_dir)
    candidates = [root / dataset_name]
    if not candidates[0].exists() and root.exists():
        candidates.append(root)

    records: list[ImageRecord] = []
    for dataset_root in candidates:
        if not dataset_root.exists():
            continue
        for path in dataset_root.rglob("*"):
            if not path.is_file() or not _is_image(path):
                continue
            rel_parts = list(path.relative_to(dataset_root).parts)
            view, view_idx = _view_from_parts(rel_parts)
            if view is None:
                continue
            label = _label_from_parts(rel_parts, view_idx, path)
            records.append(ImageRecord(
                path=path,
                dataset=dataset_name,
                label=f"{dataset_name}:{label}",
                view=view,
                explicit_split=_explicit_split(rel_parts),
            ))
        if records:
            break
    return records


def _split_labels(labels: list[str], split: str, settings: dict, seed: int) -> set[str]:
    train_ratio = float(settings.get("train_ratio", 0.8))
    val_ratio = float(settings.get("val_ratio", 0.1))
    labels = sorted(set(labels), key=lambda item: _stable_int(f"{seed}:{item}"))
    if not labels:
        return set()

    n = len(labels)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else max(0, n - n_train)
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))

    if split == "train":
        chosen = labels[:n_train]
    elif split == "val":
        chosen = labels[n_train:n_train + n_val] or labels[: min(1, n)]
    else:
        chosen = labels[n_train + n_val:] or labels[-min(1, n):]
    return set(chosen)


def _records_for_split(records: list[ImageRecord], split: str, settings: dict, seed: int) -> list[ImageRecord]:
    explicit = [record for record in records if record.explicit_split == split]
    if explicit:
        source = explicit
        labels = {record.label for record in source}
    elif split == "val":
        explicit_train = [record for record in records if record.explicit_split == "train"]
        source = explicit_train or records
        labels = _split_labels([record.label for record in source], split, settings, seed)
    elif split == "train":
        source = [record for record in records if record.explicit_split == "train"] or records
        labels = _split_labels([record.label for record in source], split, settings, seed)
    else:
        source = [record for record in records if record.explicit_split == "test"] or records
        labels = _split_labels([record.label for record in source], split, settings, seed)

    max_classes = settings.get(f"max_{split}_classes")
    if max_classes is not None:
        labels = set(sorted(labels, key=lambda item: _stable_int(f"{seed}:{split}:{item}"))[:int(max_classes)])
    return [record for record in source if record.label in labels]


def _group_by_label(records: list[ImageRecord]) -> dict[str, dict[str, list[ImageRecord]]]:
    grouped: dict[str, dict[str, list[ImageRecord]]] = {}
    for record in records:
        grouped.setdefault(record.label, {}).setdefault(record.view, []).append(record)
    return grouped


def _pick(records: list[ImageRecord], index: int) -> ImageRecord:
    return records[index % len(records)]


def _sample_count(value, records: list[ImageRecord]) -> int:
    if isinstance(value, str) and value.lower() in {"all", "*"}:
        return len(records)
    return max(1, min(int(value), len(records)))


def _make_pair_samples(records: list[ImageRecord], split: str, settings: dict, seed: int) -> list[dict]:
    grouped = _group_by_label(records)
    samples: list[dict] = []
    rng = random.Random(seed + _stable_int(split))

    for label in sorted(grouped):
        views = grouped[label]
        if not views.get("uav") or not views.get("sat"):
            continue
        if split == "train":
            per_class = _sample_count(settings.get("train_uav_samples_per_class", 2), views["uav"])
        else:
            per_class = _sample_count(
                settings.get("val_queries_per_class", settings.get("val_uav_samples_per_class", 1)),
                views["uav"],
            )
        ground_records = views.get("ground", [])
        for i in range(per_class):
            samples.append({
                "uav": _pick(views["uav"], i),
                "sat": _pick(views["sat"], rng.randrange(len(views["sat"]))),
                "ground": _pick(ground_records, i) if ground_records else None,
                "label": label,
            })
    return samples


class CombinedDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        datasets_list: list[str],
        split: str = "train",
        transform: Callable | None = None,
        dataset_settings: dict | None = None,
        seed: int = 42,
    ):
        self.root_dir = root_dir
        self.datasets_list = datasets_list
        self.split = split
        self.transform = transform
        self.dataset_settings = dataset_settings or {}
        self.seed = seed
        self.samples: list[dict] = []

        for dataset_name in datasets_list:
            settings = self.dataset_settings.get(dataset_name, {})
            records = _discover_records(root_dir, dataset_name)
            split_records = _records_for_split(records, split, settings, seed)
            self.samples.extend(_make_pair_samples(split_records, split, settings, seed))

        if not self.samples:
            raise FileNotFoundError(
                f"No {split} UAV/satellite pairs found under {root_dir} for: "
                f"{', '.join(datasets_list)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_record(self, record: ImageRecord):
        image = _read_image(record.path)
        return self.transform(image) if self.transform else image

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        item = {
            "uav": self._load_record(sample["uav"]),
            "sat": self._load_record(sample["sat"]),
            "label": sample["label"],
            "dataset": sample["uav"].dataset,
        }
        if sample.get("ground") is not None:
            item["ground"] = self._load_record(sample["ground"])
        return item


class SingleViewDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: Callable | None = None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = _read_image(record.path)
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": record.label,
            "view": record.view,
            "path": str(record.path),
        }


def _records_by_view_for_benchmark(
    root_dir: str,
    dataset_name: str,
    split: str,
    settings: dict,
    seed: int,
) -> dict[str, list[ImageRecord]]:
    records = _discover_records(root_dir, dataset_name)
    split_records = _records_for_split(records, split, settings, seed)
    grouped = _group_by_label(split_records)
    by_view = {"uav": [], "sat": [], "ground": []}

    for label, views in grouped.items():
        if views.get("uav") and views.get("sat"):
            by_view["uav"].append(views["uav"][0])
            by_view["sat"].append(views["sat"][0])
            if views.get("ground"):
                by_view["ground"].append(views["ground"][0])

    return by_view


def build_retrieval_benchmarks(
    root_dir: str,
    datasets_list: list[str],
    split: str = "test",
    transform: Callable | None = None,
    dataset_settings: dict | None = None,
    seed: int = 42,
    include_reverse: bool = False,
) -> list[dict]:
    dataset_settings = dataset_settings or {}
    benchmarks = []

    for dataset_name in datasets_list:
        settings = dataset_settings.get(dataset_name, {})
        by_view = _records_by_view_for_benchmark(root_dir, dataset_name, split, settings, seed)
        pairs = [("uav", "sat")]
        if include_reverse:
            pairs.append(("sat", "uav"))
        if by_view.get("ground"):
            pairs.append(("ground", "sat"))

        for query_view, gallery_view in pairs:
            if not by_view.get(query_view) or not by_view.get(gallery_view):
                continue
            benchmarks.append({
                "name": f"{dataset_name}_{query_view}_to_{gallery_view}_{split}",
                "query_dataset": SingleViewDataset(by_view[query_view], transform=transform),
                "gallery_dataset": SingleViewDataset(by_view[gallery_view], transform=transform),
            })

    return benchmarks
