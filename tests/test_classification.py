"""Testy jednostkowe modułu klasyfikacji: dataset, prompty i modele."""

import pytest
import torch
from pathlib import Path
from PIL import Image

from src.classification.dataset import ProductDataset, create_train_val_test_split
from src.classification.prompts import get_category_prompts
from src.common.config import MAIN_CATEGORIES, SAMPLES_METADATA_PATH, PROJECT_ROOT
from src.common.schemas import load_metadata_jsonl, CategoryPrediction


def test_prompts_generation():
    """Weryfikuje generowanie promptów dla 12 kategorii."""
    prompts_en = get_category_prompts(lang="en")
    prompts_pl = get_category_prompts(lang="pl")
    
    assert len(prompts_en) == 12
    assert len(prompts_pl) == 12
    assert all("photo" in p.lower() or "zdjęcie" in p.lower() for p in prompts_en + prompts_pl)


def test_product_dataset_multilabel():
    """Weryfikuje poprawne wczytywanie i generowanie targetu multi-hot."""
    records = load_metadata_jsonl(SAMPLES_METADATA_PATH)
    assert len(records) > 0

    ds = ProductDataset(records, multilabel=True)
    assert len(ds) == len(records)

    img, target, info = ds[0]
    assert isinstance(img, Image.Image)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (12,)
    assert target.sum().item() >= 1.0  # Przynajmniej jedna etykieta aktywna
    assert "product_id" in info
    assert "category_main" in info


def test_train_val_test_split():
    """Weryfikuje stratyfikowany podział zbioru danych."""
    records = load_metadata_jsonl(SAMPLES_METADATA_PATH)
    train_ds, val_ds, test_ds = create_train_val_test_split(
        SAMPLES_METADATA_PATH, test_size=0.2, val_size=0.2, random_state=42
    )

    total_split = len(train_ds) + len(val_ds) + len(test_ds)
    assert total_split == len(records)
    assert len(train_ds) > 0
