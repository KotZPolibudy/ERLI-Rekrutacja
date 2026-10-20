"""Moduł datasetu PyTorch do wczytywania produktów i zdjęć z obsługą single-label i multi-label."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.common.config import DEFAULT_METADATA_PATH, MAIN_CATEGORIES, PROJECT_ROOT
from src.common.schemas import ProductMetadata, load_metadata_jsonl

logger = logging.getLogger("dataset")

CATEGORY_TO_IDX: Dict[str, int] = {cat: idx for idx, cat in enumerate(MAIN_CATEGORIES)}
IDX_TO_CATEGORY: Dict[int, str] = {idx: cat for cat, idx in CATEGORY_TO_IDX.items()}


class ProductDataset(Dataset):
    """Zestaw danych PyTorch wczytujący zdjęcia produktów i etykiety na podstawie metadanych."""

    def __init__(
        self,
        records: List[ProductMetadata],
        transform: Optional[Callable[[Image.Image], Any]] = None,
        multilabel: bool = True,
        root_dir: Path = PROJECT_ROOT,
    ):
        self.records: List[ProductMetadata] = []
        self.transform = transform
        self.multilabel = multilabel
        self.root_dir = root_dir

        # Walidacja istnienia plików na dysku
        skipped_count = 0
        for r in records:
            img_path = self.root_dir / r.local_image_path
            if img_path.exists():
                self.records.append(r)
            else:
                skipped_count += 1

        if skipped_count > 0:
            logger.warning("Pominięto %d rekordów z brakującym plikiem graficznym.", skipped_count)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor, Dict[str, Any]]:
        record = self.records[idx]
        img_path = self.root_dir / record.local_image_path

        # Bezpieczne wczytanie obrazu
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            logger.error("Błąd otwarcia pliku %s: %s (generowanie pustego obrazu)", img_path, e)
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        # Generowanie targetu
        main_cat_idx = CATEGORY_TO_IDX.get(record.category_main, -1)

        if self.multilabel:
            # Wektor multi-hot o długości 12 (float32 dla BCEWithLogitsLoss)
            target = torch.zeros(len(MAIN_CATEGORIES), dtype=torch.float32)
            if main_cat_idx >= 0:
                target[main_cat_idx] = 1.0

            # Dodatkowe etykiety multi-label
            for add_cat in record.additional_categories:
                add_idx = CATEGORY_TO_IDX.get(add_cat)
                if add_idx is not None:
                    target[add_idx] = 1.0
        else:
            # Pojedynczy indeks dla standardowego CrossEntropy
            target = torch.tensor(main_cat_idx, dtype=torch.long)

        info = {
            "product_id": record.product_id,
            "title": record.title,
            "category_main": record.category_main,
            "local_image_path": record.local_image_path,
        }

        return image, target, info


def create_train_val_test_split(
    metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    multilabel: bool = True,
    transform: Optional[Callable] = None,
) -> Tuple[ProductDataset, ProductDataset, ProductDataset]:
    """Wczytuje metadane i dzieli zbiór na train, val oraz test ze stratyfikacją po kategorii głównej."""
    records = load_metadata_jsonl(metadata_path)
    if not records:
        raise ValueError(f"Brak rekordów w {metadata_path}")

    from collections import Counter

    # Pobierz etykiety kategorii głównej do stratyfikacji
    labels = [r.category_main for r in records]
    counts = Counter(labels)
    can_stratify = all(c >= 2 for c in counts.values()) and len(records) >= 10

    # Pierwszy split: train+val vs test
    train_val_records, test_records = train_test_split(
        records,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if can_stratify else None
    )

    # Drugi split: train vs val
    val_relative_size = val_size / (1.0 - test_size)
    train_val_labels = [r.category_main for r in train_val_records]
    tv_counts = Counter(train_val_labels)
    can_stratify_val = all(c >= 2 for c in tv_counts.values()) and len(train_val_records) >= 10

    train_records, val_records = train_test_split(
        train_val_records,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=train_val_labels if can_stratify_val else None
    )

    logger.info(
        "Podział zbioru: Train=%d, Val=%d, Test=%d (Łącznie=%d)",
        len(train_records), len(val_records), len(test_records), len(records)
    )

    train_ds = ProductDataset(train_records, transform=transform, multilabel=multilabel)
    val_ds = ProductDataset(val_records, transform=transform, multilabel=multilabel)
    test_ds = ProductDataset(test_records, transform=transform, multilabel=multilabel)

    return train_ds, val_ds, test_ds
