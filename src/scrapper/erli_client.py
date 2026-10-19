"""Klient HTTP do bezpiecznego i niezawodnego pobierania danych z serwisu Erli.pl.
Obsługuje bezpośrednie zapytania, sesję oraz niezawodny fallback przez curl.exe na Windowsie.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("erli_client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_URL = "https://erli.pl"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
}


class ErliClient:
    """Klient HTTP zoptymalizowany pod kątem pobierania list produktów i zdjęć z Erli."""

    def __init__(self, delay_seconds: float = 0.5, timeout: float = 20.0, max_retries: int = 3):
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = httpx.Client(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )

    def _fetch_with_curl(self, url: str) -> Optional[str]:
        """Wykonuje zapytanie za pomocą natywnego curl.exe (odpornego na Cloudflare TLS fingerprinting)."""
        try:
            cmd = [
                "curl.exe",
                "-s",
                "-L",
                "--max-time", str(int(self.timeout)),
                "-H", f"User-Agent: {DEFAULT_HEADERS['User-Agent']}",
                "-H", f"Accept: {DEFAULT_HEADERS['Accept']}",
                "-H", f"Accept-Language: {DEFAULT_HEADERS['Accept-Language']}",
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if result.returncode == 0 and result.stdout:
                return result.stdout
            else:
                logger.warning("curl.exe zwrócił kod %d dla %s", result.returncode, url)
                return None
        except Exception as e:
            logger.error("Błąd podczas wywołania curl.exe: %s", e)
            return None

    def get_html(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Pobiera kod HTML podanego adresu URL z zachowaniem rate-limitingu i fallbacku."""
        time.sleep(self.delay_seconds)
        if params:
            query_str = "&".join(f"{k}={v}" for k, v in params.items())
            full_url = f"{url}?{query_str}" if "?" not in url else f"{url}&{query_str}"
        else:
            full_url = url

        # Próba 1: httpx
        try:
            resp = self.client.get(full_url)
            if resp.status_code == 200 and len(resp.text) > 500:
                return resp.text
            elif resp.status_code == 403:
                logger.info("Status 403 z httpx dla %s – przełączanie na natywny silnik TLS...", full_url)
        except Exception as exc:
            logger.warning("Błąd httpx dla %s: %s – przełączanie na natywny silnik TLS...", full_url, exc)

        # Próba 2: curl.exe fallback
        return self._fetch_with_curl(full_url)

    def fetch_category_html(self, category_url: str, page: int = 1) -> Optional[str]:
        """Pobiera kod HTML strony kategorii z uwzględnieniem paginacji."""
        params = {"page": page} if page > 1 else None
        return self.get_html(category_url, params=params)

    def parse_products_from_html(self, html: str) -> List[Dict[str, Any]]:
        """Wyciąga listę produktów (id, nazwa, url_produktu, image_url, cena) ze strony kategorii."""
        soup = BeautifulSoup(html, "html.parser")
        products: List[Dict[str, Any]] = []
        seen_ids = set()

        # Strategia 1: Sprawdź ld+json (Itemlist)
        for script in soup.find_all("script", type="application/ld+json"):
            if not script.string:
                continue
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get("@type") == "ItemList":
                    for element in data.get("itemListElement", []):
                        item = element.get("item", {})
                        if item and item.get("@type") == "Product":
                            name = item.get("name")
                            prod_url = item.get("url", "")
                            image_url = item.get("image", "")

                            id_match = re.search(r",(\d+)$", prod_url) or re.search(r"-(\d+)$", prod_url)
                            product_id = id_match.group(1) if id_match else prod_url.split("/")[-1]

                            offers = item.get("offers", {})
                            price = None
                            if isinstance(offers, dict):
                                price = offers.get("price")

                            if product_id and product_id not in seen_ids and image_url:
                                seen_ids.add(product_id)
                                products.append({
                                    "product_id": str(product_id),
                                    "title": name,
                                    "product_url": prod_url,
                                    "image_url": image_url,
                                    "price": float(price) if price is not None else None,
                                })
            except json.JSONDecodeError:
                pass

        if products:
            return products

        # Strategia 2: Parsowanie kart produktów z HTML (fallback)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/produkt/" in href:
                id_match = re.search(r",(\d+)$", href)
                if not id_match:
                    continue
                product_id = id_match.group(1)
                if product_id in seen_ids:
                    continue

                parent = a.find_parent("div") or a
                img = parent.find("img") or a.find("img")
                if not img:
                    continue

                image_url = (
                    img.get("data-src")
                    or img.get("src")
                    or img.get("srcset", "").split(" ")[0]
                )
                if not image_url or "data:image" in image_url:
                    continue

                if image_url.startswith("//"):
                    image_url = f"https:{image_url}"
                elif image_url.startswith("/"):
                    image_url = urljoin(BASE_URL, image_url)

                title = a.get_text(strip=True) or img.get("alt", "")
                seen_ids.add(product_id)
                products.append({
                    "product_id": str(product_id),
                    "title": title,
                    "product_url": urljoin(BASE_URL, href),
                    "image_url": image_url,
                    "price": None,
                })

        return products

    def download_image(self, image_url: str, output_path: Path) -> bool:
        """Pobiera plik graficzny, waliduje jego poprawność i konwertuje/zapisuje na dysku jako JPEG.
        Zwraca True w przypadku sukcesu.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".tmp")

        # Pobieranie za pomocą curl.exe
        try:
            cmd = ["curl.exe", "-s", "-L", "--max-time", "15", "-o", str(temp_path), image_url]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0 or not temp_path.exists() or temp_path.stat().st_size < 100:
                # Próba awaryjna httpx
                resp = self.client.get(image_url, timeout=15.0)
                if resp.status_code == 200 and len(resp.content) >= 100:
                    with open(temp_path, "wb") as f:
                        f.write(resp.content)
                else:
                    logger.warning("Nieudane pobranie zdjęcia z %s", image_url)
                    if temp_path.exists():
                        temp_path.unlink()
                    return False

            # Weryfikacja i konwersja do JPEG za pomocą Pillow
            with Image.open(temp_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(output_path, "JPEG", quality=90)

            if temp_path.exists():
                temp_path.unlink()
            return True

        except Exception as exc:
            logger.error("Błąd podczas przetwarzania obrazu %s: %s", image_url, exc)
            if temp_path.exists():
                temp_path.unlink()
            return False

    def close(self) -> None:
        """Zamyka klienta HTTP."""
        self.client.close()

    def __enter__(self) -> ErliClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
