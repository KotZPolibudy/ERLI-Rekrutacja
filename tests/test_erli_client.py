"""Testy integracyjne dla klienta ErliClient."""

import pytest
from pathlib import Path
from src.scrapper.erli_client import ErliClient


def test_erli_client_init_and_fetch():
    """Weryfikuje poprawność pobierania HTML i parsowania produktów z kategorii."""
    with ErliClient(delay_seconds=0.2) as client:
        # 1. Pobierz HTML kategorii Moda
        html = client.fetch_category_html("https://erli.pl/moda,6731")
        assert html is not None, "Nie udało się pobrać HTML kategorii"
        assert len(html) > 1000

        # 3. Sparsuj produkty
        products = client.parse_products_from_html(html)
        assert len(products) > 0, "Nie znaleziono produktów na stronie kategorii"
        
        first = products[0]
        assert "product_id" in first
        assert "image_url" in first
        assert first["image_url"].startswith("http")
        assert "title" in first
        print(f"Pomyślnie sparsowano {len(products)} produktów z Erli. Przykładowy: {first['title']}")


def test_erli_client_download_image(tmp_path: Path):
    """Weryfikuje pobieranie i konwersję zdjęcia do formatu JPEG."""
    with ErliClient(delay_seconds=0.2) as client:
        html = client.fetch_category_html("https://erli.pl/moda,6731")
        assert html is not None
        products = client.parse_products_from_html(html)
        assert len(products) > 0

        # Pobierz pierwsze zdjęcie
        img_url = products[0]["image_url"]
        out_file = tmp_path / "test_download.jpg"
        
        ok = client.download_image(img_url, out_file)
        assert ok is True, f"Nie udało się pobrać i zapisać zdjęcia z {img_url}"
        assert out_file.exists()
        assert out_file.stat().st_size > 500, "Plik zdjęcia jest podejrzanie mały"
