"""Narzędzie CLI do inferencji modeli klasyfikacji kategorii:
- Klasyfikacja pojedynczego obrazu
- Klasyfikacja całego zbioru danych (z pliku metadata.jsonl)
- Obsługa modeli: siglip, clip, linear_probe
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.classification.dataset import ProductDataset, create_train_val_test_split
from src.classification.model import BaseClassifier, CLIPClassifier, LinearProbeClassifier, SigLIPClassifier
from src.common.config import DEFAULT_METADATA_PATH, PROJECT_ROOT, REPORTS_DIR, SAMPLES_METADATA_PATH
from src.common.schemas import CategoryPrediction, load_metadata_jsonl, save_jsonl

logger = logging.getLogger("predict_cli")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_classifier(model_type: str, train_metadata_path: Optional[Path] = None) -> BaseClassifier:
    """Tworzy instancję wybranego klasyfikatora."""
    model_type = model_type.lower()
    if model_type == "siglip":
        return SigLIPClassifier()
    elif model_type == "clip":
        return CLIPClassifier()
    elif model_type == "linear_probe":
        siglip = SigLIPClassifier()
        clf = LinearProbeClassifier(feature_extractor=siglip)

        meta_path = train_metadata_path or DEFAULT_METADATA_PATH
        if not meta_path.exists():
            meta_path = SAMPLES_METADATA_PATH

        logger.info("Trenowanie Linear Probe na danych z %s...", meta_path)
        train_ds, _, _ = create_train_val_test_split(meta_path, test_size=0.2, val_size=0.1)
        train_images = [PROJECT_ROOT / r.local_image_path for r in train_ds.records]
        
        # Przygotuj targety multi-hot
        targets = []
        for _, target, _ in train_ds:
            targets.append(target.numpy())
        clf.fit(train_images, targets)
        return clf
    else:
        raise ValueError(f"Nieobsługiwany typ modelu: {model_type}. Dostępne: siglip, clip, linear_probe")


def run_single_prediction(
    classifier: BaseClassifier,
    image_path: Union[str, Path],
    threshold: float = 0.35
) -> None:
    """Przeprowadza i wyświetla predykcję dla pojedynczego zdjęcia."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[BŁĄD] Plik nie istnieje: {image_path}")
        return

    pred = classifier.predict_image(image_path, threshold=threshold)

    print("\n" + "=" * 60)
    print(f"WYNIK KLASYFIKACJI ({classifier.get_model_name()})")
    print("=" * 60)
    print(f"Obraz:                {image_path.name}")
    print(f"Główna kategoria:     {pred.predicted_category} (pewność: {pred.confidence * 100:.1f}%)")
    print(f"Kategorie multi-label (>= {threshold:.2f}): {', '.join(pred.predicted_categories)}")
    print(f"Czas inferencji:      {pred.latency_ms:.2f} ms")
    print("-" * 60)
    print("TOP 3 NAJBARDZIEJ PRAWDOPODOBNE KATEGORIE:")
    for i, top in enumerate(pred.top_k_predictions[:3], 1):
        print(f"  {i}. {top.category:<22}: {top.score * 100:5.1f}%")
    print("=" * 60 + "\n")


def run_batch_evaluation(
    classifier: BaseClassifier,
    metadata_path: Path,
    output_predictions_path: Path,
    threshold: float = 0.35,
    batch_size: int = 32,
    resume: bool = True,
) -> int:
    """Klasyfikuje zbiór danych ze strumieniowym przetwarzaniem (O(1) pamięci RAM) i wznawianiem (Resuming).
    Zaprojektowany do obsługi milionów rekordów w produkcji.
    """
    from src.common.schemas import iter_metadata_jsonl

    if not metadata_path.exists():
        print(f"[BŁĄD] Plik metadanych nie istnieje: {metadata_path}")
        return 0

    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Sprawdzenie istniejących predykcji pod kątem wznawiania (Resuming)
    processed_ids = set()
    if resume and output_predictions_path.exists():
        with output_predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        p_id = record.get("product_id")
                        if p_id:
                            processed_ids.add(p_id)
                    except Exception:
                        pass
        if processed_ids:
            print(f"[Wznowienie] Wykryto {len(processed_ids)} już sklasyfikowanych produktów. Pomijam je.")

    print(f"\nRozpoczynanie strumieniowej klasyfikacji z {metadata_path}...")
    start_time = time.time()
    total_processed = len(processed_ids)
    batch_records = []
    correct_count = 0
    evaluated_with_gt = 0

    with output_predictions_path.open("a", encoding="utf-8") as out_f:
        for record in iter_metadata_jsonl(metadata_path):
            if record.product_id in processed_ids:
                continue

            img_full_path = PROJECT_ROOT / record.local_image_path
            if not img_full_path.exists():
                continue

            batch_records.append(record)

            if len(batch_records) >= batch_size:
                b_imgs = [PROJECT_ROOT / r.local_image_path for r in batch_records]
                b_ids = [r.product_id for r in batch_records]
                b_cats = [r.category_main for r in batch_records]

                preds = classifier.predict_batch(
                    images=b_imgs,
                    product_ids=b_ids,
                    true_categories=b_cats,
                    threshold=threshold,
                )

                for p in preds:
                    out_f.write(p.model_dump_json() + "\n")
                    if p.true_category:
                        evaluated_with_gt += 1
                        if p.predicted_category == p.true_category:
                            correct_count += 1

                out_f.flush()
                total_processed += len(preds)
                elapsed = time.time() - start_time
                fps = (total_processed - len(processed_ids)) / elapsed if elapsed > 0 else 0
                acc_str = f" | Acc: {(correct_count / max(1, evaluated_with_gt)) * 100:.1f}%" if evaluated_with_gt else ""
                print(f"  Przetworzono: {total_processed} produktów ({fps:.1f} FPS{acc_str})...", end="\r")
                batch_records.clear()

        # Ostatnia niepełna paczka
        if batch_records:
            b_imgs = [PROJECT_ROOT / r.local_image_path for r in batch_records]
            b_ids = [r.product_id for r in batch_records]
            b_cats = [r.category_main for r in batch_records]

            preds = classifier.predict_batch(
                images=b_imgs,
                product_ids=b_ids,
                true_categories=b_cats,
                threshold=threshold,
            )

            for p in preds:
                out_f.write(p.model_dump_json() + "\n")
                if p.true_category:
                    evaluated_with_gt += 1
                    if p.predicted_category == p.true_category:
                        correct_count += 1

            out_f.flush()
            total_processed += len(preds)

    total_time = time.time() - start_time
    newly_processed = total_processed - len(processed_ids)
    avg_latency = (total_time / newly_processed) * 1000.0 if newly_processed > 0 else 0.0

    print(f"\n" + "=" * 65)
    print(f"PODSUMOWANIE PREDYKCJI STRUMIENIOWEJ ({classifier.get_model_name()})")
    print("=" * 65)
    print(f"Łącznie w bazie wyników:  {total_processed} produktów")
    print(f"Przetworzono w tej sesji: {newly_processed} produktów")
    if evaluated_with_gt > 0:
        accuracy = (correct_count / evaluated_with_gt) * 100.0
        print(f"Trafność Top-1 (z GT):    {accuracy:.2f}% ({correct_count}/{evaluated_with_gt})")
    print(f"Średni czas / obraz:      {avg_latency:.1f} ms ({newly_processed / max(0.001, total_time):.1f} FPS)")
    print(f"Plik wynikowy:            {output_predictions_path}")
    print("=" * 65 + "\n")

    return total_processed


def main():
    parser = argparse.ArgumentParser(description="Inferencja klasyfikacji kategorii produktów Erli")
    parser.add_argument("--model", type=str, default="siglip", choices=["siglip", "clip", "linear_probe"], help="Wybór modelu")
    parser.add_argument("--image", type=str, default=None, help="Ścieżka do pojedynczego zdjęcia do sklasyfikowania")
    parser.add_argument("--metadata", type=str, default=None, help="Ścieżka do pliku metadata.jsonl dla ewaluacji batchowej")
    parser.add_argument("--threshold", type=float, default=0.35, help="Próg odcięcia dla klasyfikacji multi-label (domyślnie: 0.35)")
    parser.add_argument("--output", type=str, default=None, help="Plik wyjściowy dla predykcji JSONL")
    parser.add_argument("--batch-size", type=int, default=32, help="Rozmiar paczki przy inferencji")
    parser.add_argument("--no-resume", action="store_true", help="Jeśli ustawione, nadpisuje istniejące wyniki zamiast wznawiać")
    args = parser.parse_args()

    classifier = load_classifier(args.model)

    if args.image:
        run_single_prediction(classifier, args.image, threshold=args.threshold)
    else:
        meta_path = Path(args.metadata) if args.metadata else DEFAULT_METADATA_PATH
        if not meta_path.exists():
            meta_path = SAMPLES_METADATA_PATH

        out_path = Path(args.output) if args.output else (REPORTS_DIR / f"predictions_{args.model}.jsonl")
        run_batch_evaluation(
            classifier,
            meta_path,
            out_path,
            threshold=args.threshold,
            batch_size=args.batch_size,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
