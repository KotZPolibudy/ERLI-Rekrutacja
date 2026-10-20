"""Schematy danych i kontrakty wymiany informacji pomiędzy agentami/modułami."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ProductMetadata(BaseModel):
    """Metadane pobranego produktu i powiązanego zdjęcia."""
    product_id: str = Field(description="Unikalny identyfikator produktu w serwisie")
    title: str = Field(description="Tytuł / nazwa produktu")
    category_main: str = Field(description="Kategoria główna (poziom 0)")
    category_sub1: Optional[str] = Field(default=None, description="Podkategoria 1. stopnia")
    category_path: List[str] = Field(default_factory=list, description="Pełna ścieżka w drzewie kategorii")
    additional_categories: List[str] = Field(
        default_factory=list,
        description="Dodatkowe kategorie (dla scenariuszy multi-label, np. Moda + Sport)"
    )
    image_url: str = Field(description="Oryginalny URL zdjęcia produktu")
    local_image_path: str = Field(description="Względna ścieżka do zapisanego pliku graficznego")
    price: Optional[float] = Field(default=None, description="Cena produktu")
    currency: str = Field(default="PLN", description="Waluta ceny")
    downloaded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Znacznik czasu pobrania w formacie ISO"
    )


class CategoryScore(BaseModel):
    """Pojedyncza para kategoria i prawdopodobieństwo z predykcji top-k."""
    category: str = Field(description="Nazwa kategorii")
    score: float = Field(description="Prawdopodobieństwo / pewność modelu")


class CategoryPrediction(BaseModel):
    """Wynik predykcji klasyfikatora kategorii dla pojedynczego produktu/obrazu."""
    product_id: str
    image_path: str
    true_category: Optional[str] = None
    true_categories: List[str] = Field(default_factory=list, description="Rzeczywiste kategorie w scenariuszu multi-label")
    predicted_category: str
    predicted_categories: List[str] = Field(default_factory=list, description="Przewidziane kategorie dla multi-label")
    is_multilabel: bool = Field(default=False, description="Czy wynik dotyczy ewaluacji multi-label")
    confidence: float
    top_k_predictions: List[CategoryScore] = Field(
        default_factory=list,
        description="Lista predykcji top-k posortowana malejąco"
    )
    latency_ms: float = Field(description="Czas przetwarzania pojedynczego obrazu w milisekundach")


class BBox(BaseModel):
    """Pojedynczy bounding box dla detekcji elementu ubioru."""
    x1: float = Field(description="Współrzędna lewej krawędzi (piksele lub znormalizowane 0-1)")
    y1: float = Field(description="Współrzędna górnej krawędzi")
    x2: float = Field(description="Współrzędna prawej krawędzi")
    y2: float = Field(description="Współrzędna dolnej krawędzi")
    label: str = Field(description="Nazwa wykrytego elementu (np. dress, jacket, pants, shoes)")
    confidence: float = Field(description="Poziom pewności detekcji (0-1)")


class FashionDetectionResult(BaseModel):
    """Wynik detekcji elementów ubioru dla danego zdjęcia."""
    product_id: str
    image_path: str
    detections: List[BBox] = Field(default_factory=list)
    latency_ms: float = Field(description="Czas wykonania detekcji w ms")


# ============================================================================
# Funkcje pomocnicze do odczytu / zapisu formatu JSONL
# ============================================================================

def save_jsonl(records: List[Union[ProductMetadata, CategoryPrediction, FashionDetectionResult, Dict[str, Any]]], output_path: Union[str, Path]) -> None:
    """Zapisuje listę obiektów (Pydantic lub dict) do pliku w formacie JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            if isinstance(record, BaseModel):
                line = record.model_dump_json()
            else:
                line = json.dumps(record, ensure_ascii=False)
            f.write(line + "\n")


def load_metadata_jsonl(input_path: Union[str, Path]) -> List[ProductMetadata]:
    """Wczytuje listę ProductMetadata z pliku JSONL."""
    return list(iter_metadata_jsonl(input_path))


def iter_metadata_jsonl(input_path: Union[str, Path]):
    """Generator strumieniowy czytający rekordy ProductMetadata z pliku JSONL linia po linii (stała pamięć O(1)).
    Kluczowy dla baz danych o skali setek tysięcy i milionów produktów.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        return
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    yield ProductMetadata.model_validate(data)
                except Exception:
                    continue
