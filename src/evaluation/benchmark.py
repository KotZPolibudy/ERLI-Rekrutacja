"""Skrypt CLI do uruchomienia pełnego benchmarku modeli klasyfikacji:
- SigLIP (Google, sigmoid multi-label)
- CLIP (OpenAI, softmax baseline)
- Linear Probe (One-vs-Rest na cechach SigLIP)

Uruchomienie:
    python -m src.evaluation.benchmark --samples-per-class 25 --output-dir reports
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Zapewnienie kodowania UTF-8 na Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.classification.dataset import create_train_val_test_split, load_metadata_jsonl
from src.classification.model import CLIPClassifier, LinearProbeClassifier, SigLIPClassifier
from src.common.config import MAIN_CATEGORIES, PROJECT_ROOT, RAW_DIR
from src.common.schemas import CategoryPrediction, ProductMetadata
from src.evaluation.metrics import (
    EvaluationMetrics,
    analyze_errors_and_multilabel,
    compute_metrics,
    generate_confusion_matrix_ascii,
    generate_markdown_comparison_report,
    plot_and_save_confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("benchmark")


def save_jsonl(predictions: List[CategoryPrediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(p.model_dump_json() + "\n")


def build_evaluation_sample(
    metadata_path: Path,
    samples_per_class: int = 25,
) -> List[ProductMetadata]:
    """Wybiera zrównoważoną próbkę produktów z metadata.jsonl do rzetelnego i szybkiego benchmarku."""
    records = load_metadata_jsonl(metadata_path)
    if not records:
        raise ValueError(f"Brak rekordów w {metadata_path}")

    by_cat = {cat: [] for cat in MAIN_CATEGORIES}
    for r in records:
        if r.category_main in by_cat:
            img_p = PROJECT_ROOT / r.local_image_path
            if img_p.exists():
                by_cat[r.category_main].append(r)

    import random
    rng = random.Random(42)
    eval_sample: List[ProductMetadata] = []
    for cat in MAIN_CATEGORIES:
        pool = by_cat[cat]
        count = min(samples_per_class, len(pool))
        sample = rng.sample(pool, count) if len(pool) >= count else pool
        eval_sample.extend(sample)
        logger.info("Kategoria %-24s: wybrano %d produktów do ewaluacji", cat, len(sample))

    logger.info("Łączny rozmiar próbki ewaluacyjnej: %d produktów.", len(eval_sample))
    return eval_sample


def main():
    parser = argparse.ArgumentParser(description="Benchmark klasyfikatorów kategorii Erli")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=RAW_DIR / "metadata.jsonl",
        help="Ścieżka do metadata.jsonl",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=25,
        help="Liczba próbek z każdej kategorii do testu (domyślnie 25 x 12 = 300 produktów)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Katalog na raporty, wykresy i wyniki",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Rozmiar paczki przy inferencji",
    )
    parser.add_argument(
        "--include-linear-probe",
        action="store_true",
        help="Czy włączyć do porównania Linear Probe (wymaga fit na części danych)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Przygotowanie danych ewaluacyjnych
    eval_records = build_evaluation_sample(args.metadata, args.samples_per_class)
    eval_images = [PROJECT_ROOT / r.local_image_path for r in eval_records]
    eval_ids = [r.product_id for r in eval_records]
    eval_true_cats = [r.category_main for r in eval_records]

    siglip_clf = SigLIPClassifier(prompt_lang="en")
    clip_clf = CLIPClassifier(prompt_lang="en")

    models_to_test: List[Tuple[str, Any]] = [
        ("SigLIP (Multi-Label Sigmoid)", siglip_clf),
        ("CLIP (Softmax Baseline)", clip_clf),
    ]

    if args.include_linear_probe:
        logger.info("\n" + "=" * 60)
        logger.info("TRENOWANIE LINEAR PROBE NA CECHACH SIGLIP")
        logger.info("=" * 60)
        import numpy as np
        import random
        eval_id_set = {r.product_id for r in eval_records}
        all_records = load_metadata_jsonl(args.metadata)
        train_pool = [r for r in all_records if r.product_id not in eval_id_set and (PROJECT_ROOT / r.local_image_path).exists()]

        rng_train = random.Random(42)
        by_cat_train = {c: [] for c in MAIN_CATEGORIES}
        for r in train_pool:
            if r.category_main in by_cat_train:
                by_cat_train[r.category_main].append(r)

        train_records = []
        for c in MAIN_CATEGORIES:
            pool = by_cat_train[c]
            cnt = min(20, len(pool))
            train_records.extend(rng_train.sample(pool, cnt) if len(pool) >= cnt else pool)

        train_imgs = [PROJECT_ROOT / r.local_image_path for r in train_records]
        cat2idx = {cat: i for i, cat in enumerate(MAIN_CATEGORIES)}
        targets = np.zeros((len(train_records), len(MAIN_CATEGORIES)), dtype=int)
        for i, r in enumerate(train_records):
            targets[i, cat2idx[r.category_main]] = 1

        lp_clf = LinearProbeClassifier(feature_extractor=siglip_clf)
        lp_clf.fit(train_imgs, targets)
        models_to_test.append(("Linear Probe (SigLIP Features)", lp_clf))

    all_metrics: List[EvaluationMetrics] = []

    for model_label, clf in models_to_test:
        logger.info("\n" + "=" * 60)
        logger.info("ROZPOCZĘCIE EWALUACJI DLA: %s", model_label)
        logger.info("=" * 60)

        preds: List[CategoryPrediction] = []
        for i in range(0, len(eval_images), args.batch_size):
            b_imgs = eval_images[i : i + args.batch_size]
            b_ids = eval_ids[i : i + args.batch_size]
            b_cats = eval_true_cats[i : i + args.batch_size]
            b_preds = clf.predict_batch(b_imgs, b_ids, b_cats)
            preds.extend(b_preds)
            print(f"  [{model_label}] Postęp: {len(preds)}/{len(eval_images)}...", end="\r")

        print()
        if "Linear" in model_label:
            slug = "linear_probe"
        elif "SigLIP" in model_label:
            slug = "siglip"
        else:
            slug = "clip"
        pred_path = args.output_dir / f"predictions_{slug}.jsonl"
        save_jsonl(preds, pred_path)

        # Oblicz metryki
        metrics = compute_metrics(preds, model_name=model_label)
        all_metrics.append(metrics)

        # Generuj macierz pomyłek ASCII i PNG
        cm_ascii = generate_confusion_matrix_ascii(preds)
        print("\n" + cm_ascii + "\n")
        cm_png_path = args.output_dir / f"confusion_matrix_{slug}.png"
        plot_and_save_confusion_matrix(preds, cm_png_path, model_name=model_label)

        # Analiza błędów
        err_analysis = analyze_errors_and_multilabel(preds)
        err_path = args.output_dir / f"error_analysis_{slug}.json"
        err_path.write_text(json.dumps(err_analysis, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Zakończono ewaluację %s: Top-1 Acc: %.2f%%, Macro F1: %.2f%%, Latencja: %.1f ms",
                    model_label, metrics.top1_accuracy * 100, metrics.macro_f1 * 100, metrics.avg_latency_ms)

    # 3. Zapisanie całościowego raportu podsumowującego
    summary_path = args.output_dir / "benchmark_summary.json"
    summary_data = [m.to_dict() for m in all_metrics]
    summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    md_report_path = args.output_dir / "classification_report.md"
    generate_markdown_comparison_report(all_metrics, md_report_path)

    print("\n" + "=" * 70)
    print("BENCHMARK UKOŃCZONY POMYŚLNIE!")
    print(f"Wyniki zapisano w katalogu: {args.output_dir}")
    print(f"1. Raport Markdown:       {md_report_path}")
    print(f"2. Podsumowanie JSON:     {summary_path}")
    print(f"3. Macierze pomyłek PNG:  {args.output_dir}/confusion_matrix_*.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
