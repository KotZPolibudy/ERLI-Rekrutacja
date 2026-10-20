"""Moduł próbkujący i pobierający zdjęcia produktów oraz metadane z serwisu Erli.
Zapisuje zdjęcia do data/raw/<kategoria_glowna>/<podkategoria>/<product_id>.jpg
oraz metadane do data/raw/metadata.jsonl z zachowaniem kontraktu danych.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.common.config import DATA_DIR, DEFAULT_METADATA_PATH, MAIN_CATEGORIES, RAW_DIR
from src.common.schemas import ProductMetadata, load_metadata_jsonl, save_jsonl
from src.scrapper.category_tree import get_full_category_tree, sanitize_folder_name
from src.scrapper.erli_client import ErliClient

logger = logging.getLogger("collector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ProductCollector:
    """Kolektor reprezentatywnej próbki produktów z serwisu Erli."""

    def __init__(
        self,
        target_per_category: int = 35,
        max_per_subcategory: int = 7,
        output_metadata_path: Path = DEFAULT_METADATA_PATH,
        delay_seconds: float = 0.3,
    ):
        self.target_per_category = target_per_category
        self.max_per_subcategory = max_per_subcategory
        self.metadata_path = output_metadata_path
        self.client = ErliClient(delay_seconds=delay_seconds)
        self.existing_metadata: List[ProductMetadata] = []
        self.collected_product_ids: Set[str] = set()

        self._load_existing_state()

    def _load_existing_state(self) -> None:
        """Ładuje istniejące metadane, aby zapobiec ponownemu pobieraniu tych samych produktów."""
        if self.metadata_path.exists():
            self.existing_metadata = load_metadata_jsonl(self.metadata_path)
            self.collected_product_ids = {m.product_id for m in self.existing_metadata}
            logger.info("Załadowano %d istniejących rekordów z %s", len(self.existing_metadata), self.metadata_path)

    def count_collected_for_category(self, category_main: str) -> int:
        """Zwraca liczbę dotychczas zebranych produktów dla danej kategorii głównej."""
        return sum(1 for m in self.existing_metadata if m.category_main == category_main)

    def collect_samples_for_category(
        self, category_main: str, subcategories: List[Dict[str, str]]
    ) -> List[ProductMetadata]:
        """Pobiera zbalansowaną próbkę produktów dla wybranej kategorii głównej z jej podkategorii."""
        current_count = self.count_collected_for_category(category_main)
        needed = self.target_per_category - current_count
        if needed <= 0:
            logger.info("Kategoria '%s' ma już wymaganą liczbę próbek (%d/%d).", category_main, current_count, self.target_per_category)
            return []

        logger.info("Pobieranie próbek dla '%s' (potrzeba jeszcze %d produktów)...", category_main, needed)
        new_records: List[ProductMetadata] = []

        # Iteruj po podkategoriach w celu zapewnienia różnorodności próbki
        for subcat in subcategories:
            if len(new_records) >= needed:
                break

            subcat_name = subcat["name"]
            subcat_clean = subcat["clean_name"]
            subcat_url = subcat["url"]

            logger.info("  -> Skanowanie podkategorii: %s", subcat_name)
            html = self.client.fetch_category_html(subcat_url, page=1)
            if not html:
                continue

            products = self.client.parse_products_from_html(html)
            sub_collected = 0

            for prod in products:
                if len(new_records) >= needed:
                    break
                if sub_collected >= self.max_per_subcategory:
                    break

                prod_id = prod["product_id"]
                if prod_id in self.collected_product_ids:
                    continue

                # Ścieżka docelowa dla zdjęcia
                rel_img_path = Path("data") / "raw" / category_main / subcat_clean / f"{prod_id}.jpg"
                abs_img_path = RAW_DIR / category_main / subcat_clean / f"{prod_id}.jpg"

                # Pobierz zdjęcie
                if not abs_img_path.exists():
                    success = self.client.download_image(prod["image_url"], abs_img_path)
                    if not success:
                        continue

                # Utwórz rekord metadanych (ProductMetadata)
                record = ProductMetadata(
                    product_id=prod_id,
                    title=prod["title"],
                    category_main=category_main,
                    category_sub1=subcat_name,
                    category_path=[category_main, subcat_name],
                    additional_categories=[],
                    image_url=prod["image_url"],
                    local_image_path=str(rel_img_path).replace("\\", "/"),
                    price=prod.get("price"),
                    currency="PLN",
                )

                new_records.append(record)
                self.existing_metadata.append(record)
                self.collected_product_ids.add(prod_id)
                sub_collected += 1

                # Dopisz rekord natychmiastowo w trybie append (bezpieczne dla równoczesnego odczytu i szybkie)
                with open(self.metadata_path, "a", encoding="utf-8") as f:
                    f.write(record.model_dump_json() + "\n")

        logger.info("Zakończono pobieranie dla '%s'. Zebrano nowych: %d (Łącznie: %d/%d)",
                    category_main, len(new_records), self.count_collected_for_category(category_main), self.target_per_category)
        return new_records

    def collect_all_categories(self) -> None:
        """Pobiera próbki dla wszystkich 12 kategorii głównych z zachowaniem balansu."""
        tree = get_full_category_tree()
        start_time = time.time()

        print("\n" + "=" * 60)
        print(f"ROZPOCZĘCIE PRÓBKOWANIA KATALOGU ERLI (Cel: {self.target_per_category} prod/kategorię)")
        print("=" * 60)

        for category_main in MAIN_CATEGORIES:
            subcats = tree.get(category_main, [])
            if not subcats:
                logger.warning("Brak zdefiniowanych podkategorii dla %s", category_main)
                continue
            self.collect_samples_for_category(category_main, subcats)

        total_time = time.time() - start_time
        self.print_summary(total_time)

    def print_summary(self, elapsed_seconds: float) -> None:
        """Wyświetla podsumowanie zebranych danych."""
        print("\n" + "=" * 60)
        print("PODSUMOWANIE POBRANYCH DANYCH")
        print("=" * 60)
        total = len(self.existing_metadata)
        for cat in MAIN_CATEGORIES:
            c = sum(1 for m in self.existing_metadata if m.category_main == cat)
            print(f" - {cat:<22}: {c:3d} produktów")
        print("-" * 60)
        print(f"ŁĄCZNIE PRODUKTÓW: {total}")
        print(f"CZAS WYKONANIA:    {elapsed_seconds:.1f} sekund")
        print(f"PLIK METADANYCH:   {self.metadata_path}")
        print("=" * 60 + "\n")

    def close(self) -> None:
        self.client.close()


def main():
    parser = argparse.ArgumentParser(description="Pobieranie reprezentatywnej próbki produktów i zdjęć z Erli.pl")
    parser.add_argument("--target-per-category", type=int, default=35, help="Docelowa liczba produktów na kategorię główną (domyślnie: 35)")
    parser.add_argument("--max-per-subcat", type=int, default=7, help="Maksymalna liczba produktów z jednej podkategorii (domyślnie: 7)")
    parser.add_argument("--delay", type=float, default=0.25, help="Odstęp między zapytaniami w sekundach")
    args = parser.parse_args()

    collector = ProductCollector(
        target_per_category=args.target_per_category,
        max_per_subcategory=args.max_per_subcat,
        delay_seconds=args.delay,
    )
    try:
        collector.collect_all_categories()
    finally:
        collector.close()


if __name__ == "__main__":
    main()
