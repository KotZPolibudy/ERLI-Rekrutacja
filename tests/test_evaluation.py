"""Testy jednostkowe modułu ewaluacji (src/evaluation/metrics.py)."""

import pytest
from src.common.schemas import CategoryPrediction, CategoryScore
from src.evaluation.metrics import (
    compute_metrics,
    generate_confusion_matrix_ascii,
    analyze_errors_and_multilabel,
)


def test_compute_metrics_single_and_multilabel():
    # Przygotuj mockowe predykcje
    predictions = [
        CategoryPrediction(
            product_id="p1",
            image_path="test1.jpg",
            true_category="Moda",
            true_categories=["Moda"],
            predicted_category="Moda",
            predicted_categories=["Moda", "Sport i turystyka"],
            is_multilabel=True,
            confidence=0.85,
            top_k_predictions=[
                CategoryScore(category="Moda", score=0.85),
                CategoryScore(category="Sport i turystyka", score=0.60),
                CategoryScore(category="Elektronika", score=0.05),
            ],
            latency_ms=25.0,
        ),
        CategoryPrediction(
            product_id="p2",
            image_path="test2.jpg",
            true_category="Elektronika",
            true_categories=["Elektronika"],
            predicted_category="Elektronika",
            predicted_categories=["Elektronika"],
            is_multilabel=False,
            confidence=0.92,
            top_k_predictions=[
                CategoryScore(category="Elektronika", score=0.92),
                CategoryScore(category="Firma", score=0.10),
                CategoryScore(category="Dom i Ogród", score=0.05),
            ],
            latency_ms=30.0,
        ),
        CategoryPrediction(
            product_id="p3",
            image_path="test3.jpg",
            true_category="Dom i Ogród",
            true_categories=["Dom i Ogród"],
            predicted_category="Firma",
            predicted_categories=["Firma"],
            is_multilabel=False,
            confidence=0.70,
            top_k_predictions=[
                CategoryScore(category="Firma", score=0.70),
                CategoryScore(category="Dom i Ogród", score=0.55),
                CategoryScore(category="Motoryzacja", score=0.12),
            ],
            latency_ms=20.0,
        ),
    ]

    metrics = compute_metrics(predictions, model_name="MockModel")

    # 2 poprawne na 3 próbki = 66.67%
    assert round(metrics.top1_accuracy, 2) == 0.67
    # Wszystkie 3 mają true_category w top-2 -> top3_accuracy = 1.0 (100%)
    assert metrics.top3_accuracy == 1.0
    assert metrics.multilabel_samples_count == 1
    assert metrics.total_samples == 3
    assert metrics.avg_latency_ms == 25.0
    assert metrics.throughput_fps > 0

    # Sprawdź analizę błędów
    analysis = analyze_errors_and_multilabel(predictions)
    assert analysis["total_errors"] == 1
    assert analysis["total_multilabel"] == 1
    assert analysis["sample_errors"][0]["product_id"] == "p3"

    # Sprawdź format macierzy pomyłek ASCII
    ascii_cm = generate_confusion_matrix_ascii(predictions)
    assert "Rzeczywista \\ Predykcja" in ascii_cm
    assert "Legenda kolumn" in ascii_cm
