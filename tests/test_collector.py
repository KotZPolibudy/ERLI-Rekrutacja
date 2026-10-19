"""Testy poprawności pobranych danych w data/raw/metadata.jsonl."""

from pathlib import Path
from PIL import Image

from src.common.config import DEFAULT_METADATA_PATH, MAIN_CATEGORIES, PROJECT_ROOT
from src.common.schemas import ProductMetadata, load_metadata_jsonl


def test_collected_raw_metadata():
    """Weryfikuje poprawność pliku metadata.jsonl oraz fizyczną obecność zdjęć na dysku."""
    assert DEFAULT_METADATA_PATH.exists(), "Brak pliku metadata.jsonl"
    records = load_metadata_jsonl(DEFAULT_METADATA_PATH)
    assert len(records) > 0, "Brak rekordów w metadata.jsonl"

    for r in records:
        assert isinstance(r, ProductMetadata)
        assert r.category_main in MAIN_CATEGORIES
        assert r.product_id
        assert r.image_url.startswith("http")

        # Sprawdź istnienie i poprawność pliku graficznego
        img_path = PROJECT_ROOT / r.local_image_path
        assert img_path.exists(), f"Plik {img_path} nie istnieje na dysku!"
        assert img_path.stat().st_size > 500, f"Plik {img_path} jest za mały!"

        # Sprawdź czy PIL otwiera obrazek bez błędów
        with Image.open(img_path) as img:
            assert img.format in ("JPEG", "PNG", "WEBP")
            assert img.width > 10 and img.height > 10
