"""Moduł odpowiedzialny za mapowanie drzewa kategorii Erli (kategorie główne i podkategorie 1. stopnia)
oraz przygotowanie struktury folderów na dysku.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
import httpx
from bs4 import BeautifulSoup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.common.config import DATA_DIR, RAW_DIR, MAIN_CATEGORIES
from src.scrapper.erli_client import ErliClient


# URL mapy witryny z kategoriami
SITEMAP_CATEGORIES_URL = "https://erli.pl/mapa-strony/kategorie"
CATEGORY_TREE_CACHE_FILE = DATA_DIR / "category_tree.json"

# Mapowanie 12 kategorii głównych na ich slug i ID w Erli
MAIN_CATEGORY_MAP: Dict[str, Dict[str, str]] = {
    "Dom i Ogród": {"slug": "dom-i-ogrod,1", "url": "https://erli.pl/mapa-strony/kategoria/dom-i-ogrod,1"},
    "Dziecko": {"slug": "dziecko,1014", "url": "https://erli.pl/mapa-strony/kategoria/dziecko,1014"},
    "Elektronika": {"slug": "elektronika,1899", "url": "https://erli.pl/mapa-strony/kategoria/elektronika,1899"},
    "Firma": {"slug": "firma,4282", "url": "https://erli.pl/mapa-strony/kategoria/firma,4282"},
    "Kolekcje i sztuka": {"slug": "kolekcje-i-sztuka,4949", "url": "https://erli.pl/mapa-strony/kategoria/kolekcje-i-sztuka,4949"},
    "Kultura i rozrywka": {"slug": "kultura-i-rozrywka,5450", "url": "https://erli.pl/mapa-strony/kategoria/kultura-i-rozrywka,5450"},
    "Moda": {"slug": "moda,6731", "url": "https://erli.pl/mapa-strony/kategoria/moda,6731"},
    "Motoryzacja": {"slug": "motoryzacja,7107", "url": "https://erli.pl/mapa-strony/kategoria/motoryzacja,7107"},
    "Sport i turystyka": {"slug": "sport-i-turystyka,9017", "url": "https://erli.pl/mapa-strony/kategoria/sport-i-turystyka,9017"},
    "Supermarket": {"slug": "supermarket,10299", "url": "https://erli.pl/mapa-strony/kategoria/supermarket,10299"},
    "Uroda": {"slug": "uroda,10884", "url": "https://erli.pl/mapa-strony/kategoria/uroda,10884"},
    "Zdrowie": {"slug": "zdrowie,11223", "url": "https://erli.pl/mapa-strony/kategoria/zdrowie,11223"},
}


def sanitize_folder_name(name: str) -> str:
    """Czyści nazwę kategorii ze znaków niedozwolonych w systemie plików Windows."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    sanitized = sanitized.strip().replace("\n", " ").replace("\r", "")
    return re.sub(r"\s+", " ", sanitized)


def fetch_subcategories_for_main(category_name: str, client: Optional[ErliClient] = None) -> List[Dict[str, str]]:
    """Pobiera listę podkategorii 1. stopnia dla danej kategorii głównej z mapy strony Erli."""
    cat_info = MAIN_CATEGORY_MAP.get(category_name)
    if not cat_info:
        raise ValueError(f"Nieznana kategoria główna: {category_name}")

    close_client = False
    if client is None:
        client = ErliClient(delay_seconds=0.3)
        close_client = True

    try:
        html = client.get_html(cat_info["url"])
        if not html:
            print(f"[OSTRZEŻENIE] Brak zawartości HTML dla {category_name}")
            return []

        soup = BeautifulSoup(html, "html.parser")
        subcategories: List[Dict[str, str]] = []
        seen_names = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            name = a.get_text(strip=True)
            if (
                href.startswith("/")
                and not href.startswith(("/konto", "/koszyk", "/moje-konto", "/mapa-strony", "/regulamin"))
                and re.search(r",\d+$", href)
                and name
                and name not in seen_names
                and not name.lower().startswith(("zobacz", "wróć", "moje"))
            ):
                seen_names.add(name)
                full_url = f"https://erli.pl{href}" if href.startswith("/") else href
                subcategories.append({
                    "name": name,
                    "clean_name": sanitize_folder_name(name),
                    "url": full_url,
                    "path": href,
                })
        return subcategories
    finally:
        if close_client:
            client.close()


def get_full_category_tree(force_refresh: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """Zwraca pełne drzewo kategorii (główne -> podkategorie 1. stopnia).
    Wykorzystuje plik cache data/category_tree.json jeśli istnieje.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not force_refresh and CATEGORY_TREE_CACHE_FILE.exists():
        with open(CATEGORY_TREE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Pobieranie aktualnego drzewa kategorii z serwisu Erli...")
    tree: Dict[str, List[Dict[str, str]]] = {}
    with ErliClient(delay_seconds=0.3) as client:
        for main_cat in MAIN_CATEGORIES:
            subs = fetch_subcategories_for_main(main_cat, client=client)
            tree[main_cat] = subs
            print(f"  - {main_cat}: znaleziono {len(subs)} podkategorii")

    with open(CATEGORY_TREE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    return tree


def prepare_data_directories(limit_subcategories: int = 10) -> None:
    """Tworzy katalogi kategorii głównych oraz podkategorii 1. stopnia w data/raw.
    Domyślnie tworzy podfoldery dla pierwszych `limit_subcategories` podkategorii z każdej kategorii.
    """
    tree = get_full_category_tree()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    total_created = 0
    for main_cat, subcats in tree.items():
        main_dir = RAW_DIR / main_cat
        main_dir.mkdir(parents=True, exist_ok=True)
        
        # Wybierz podkategorie (do limitu)
        selected_subcats = subcats[:limit_subcategories] if limit_subcategories > 0 else subcats
        for sub in selected_subcats:
            sub_folder = main_dir / sub["clean_name"]
            sub_folder.mkdir(parents=True, exist_ok=True)
            total_created += 1
            
    print(f"Przygotowano strukturę katalogów w {RAW_DIR} (utworzono {total_created} podfolderów).")


if __name__ == "__main__":
    get_full_category_tree(force_refresh=True)
    prepare_data_directories(limit_subcategories=8)
