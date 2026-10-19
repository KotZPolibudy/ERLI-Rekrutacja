"""Testy jednostkowe schematów i poprawności danych mockowych."""

import json
from pathlib import Path
from src.common.config import PROJECT_ROOT, SAMPLES_DIR, SAMPLES_METADATA_PATH, MAIN_CATEGORIES
from src.common.schemas import (
    ProductMetadata,
    CategoryScore,
    CategoryPrediction,
    FashionDetectionResult,
    BBox,
    load_metadata_jsonl,
    save_jsonl,
)


def test_main_categories_count():
    """Sprawdza, czy zdefiniowano dokładnie 12 kategorii głównych Erli."""
    assert len(MAIN_CATEGORIES) == 12
    assert "Moda" in MAIN_CATEGORIES
    assert "Elektronika" in MAIN_CATEGORIES


def test_load_samples_metadata():
    """Sprawdza, czy plik samples_metadata.jsonl jest poprawnie wczytywany i walidowany."""
    assert SAMPLES_METADATA_PATH.exists(), f"Brak pliku {SAMPLES_METADATA_PATH}"
    
    records = load_metadata_jsonl(SAMPLES_METADATA_PATH)
    assert len(records) >= 5
    
    for record in records:
        assert isinstance(record, ProductMetadata)
        assert record.product_id
        assert record.category_main in MAIN_CATEGORIES
        assert record.local_image_path
        
        # Sprawdzenie czy kategoria podrzędna jest podana
        if record.category_sub1:
            assert isinstance(record.category_sub1, str)


def test_category_prediction_schema():
    """Sprawdza serializację i walidację CategoryPrediction."""
    pred = CategoryPrediction(
        product_id="test_01",
        image_path="data/samples/images/Moda/sample_001.jpg",
        true_category="Moda",
        predicted_category="Moda",
        confidence=0.92,
        top_k_predictions=[
            {"category": "Moda", "score": 0.92},
            {"category": "Uroda", "score": 0.05},
        ],
        latency_ms=45.2,
    )
    dumped = pred.model_dump()
    assert dumped["predicted_category"] == "Moda"
    assert dumped["confidence"] == 0.92


def test_fashion_detection_schema():
    """Sprawdza serializację i walidację FashionDetectionResult."""
    det_result = FashionDetectionResult(
        product_id="test_02",
        image_path="data/samples/images/Moda/sample_005.jpg",
        detections=[
            BBox(x1=50.0, y1=80.0, x2=220.0, y2=450.0, label="dress", confidence=0.88),
            BBox(x1=70.0, y1=460.0, x2=140.0, y2=520.0, label="shoes", confidence=0.75),
        ],
        latency_ms=120.5,
    )
    dumped = det_result.model_dump()
    assert len(dumped["detections"]) == 2
    assert dumped["detections"][0]["label"] == "dress"
