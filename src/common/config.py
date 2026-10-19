"""Konfiguracja ścieżek i stałych dla projektu ERLI-Rekrut."""

from pathlib import Path
from typing import List

# Główne ścieżki projektu
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_DIR = DATA_DIR / "samples"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 12 oficjalnych kategorii głównych serwisu Erli
MAIN_CATEGORIES: List[str] = [
    "Dom i Ogród",
    "Dziecko",
    "Elektronika",
    "Firma",
    "Kolekcje i sztuka",
    "Kultura i rozrywka",
    "Moda",
    "Motoryzacja",
    "Sport i turystyka",
    "Supermarket",
    "Uroda",
    "Zdrowie",
]

# Domyślny plik metadanych
DEFAULT_METADATA_PATH = RAW_DIR / "metadata.jsonl"
SAMPLES_METADATA_PATH = SAMPLES_DIR / "samples_metadata.jsonl"


def ensure_directories() -> None:
    """Tworzy kluczowe katalogi projektu, jeśli nie istnieją."""
    for directory in [DATA_DIR, RAW_DIR, SAMPLES_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Upewnij się, że foldery kategorii głównych w data/raw istnieją
    for category in MAIN_CATEGORIES:
        (RAW_DIR / category).mkdir(parents=True, exist_ok=True)
