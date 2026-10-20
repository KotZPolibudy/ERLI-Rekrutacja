"""Skrypt CLI do uruchomienia benchmarku modeli klasyfikacji (SigLIP, CLIP, Linear Probe)
oraz ewaluacji predykcji wygenerowanych przez kolejkę RabbitMQ lub inferencję masową.

Funkcje:
- Obsługa akceleracji GPU (CUDA) oraz CPU (--device cuda / cpu / auto)
- Zapis wyników do unikalnego podkatalogu z oznaczeniem urządzenia i znacznikiem czasu:
  reports/benchmark_{device}_{YYYYMMDD_HHMMSS}/
- Odporność na działający w tle scrapper (bezpieczne pomijanie plików w trakcie zapisu)
- Tryb ewaluacji z gotowego pliku predykcji JSONL (--evaluate-file)
- Skalowalność do setek tysięcy próbek (strumieniowanie)

Przykłady uruchomienia:
    # 1. Benchmark porównawczy na GPU (np. 50 próbek na kategorię = 600 produktów)
    python -m src.evaluation.benchmark --device cuda --samples-per-class 50 --include-linear-probe

    # 2. Ewaluacja predykcji z kolejki RabbitMQ lub masowej predykcji:
    python -m src.evaluation.benchmark --evaluate-file reports/predictions_siglip.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Zapewnienie kodowania UTF-8 na Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from src.classification.model import CLIPClassifier, LinearProbeClassifier, SigLIPClassifier
from src.common.config import MAIN_CATEGORIES, PROJECT_ROOT, RAW_DIR, REPORTS_DIR
from src.common.schemas import CategoryPrediction, ProductMetadata, iter_metadata_jsonl
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


def is_image_valid_and_ready(image_path: Path) -> bool:
    """Sprawdza czy plik obrazu istnieje, jest kompletny i nie jest w trakcie zapisu przez scrapper."""
    try:
        if not image_path.exists():
            return False
        # Minimalny poprawny plik JPEG ma przynajmniej kilkaset bajtów
        if image_path.stat().st_size < 500:
            return False
        return True
    except (OSError, PermissionError):
        # Plik może być w trakcie zapisu przez proces pobierający
        return False


def build_evaluation_sample(
    metadata_path: Path,
    samples_per_class: Optional[int] = 50,
    sample_ratio: Optional[float] = None,
    seed: int = 42,
) -> List[ProductMetadata]:
    """Wybiera zrównoważoną próbkę produktów z metadata.jsonl.
    Obsługuje limit per klasa, ratio z całości lub wszystkie próbki.
    Bezpiecznie ignoruje pliki, które scrapper w tle może właśnie zapisywać.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Plik metadanych nie istnieje: {metadata_path}")

    by_cat = {cat: [] for cat in MAIN_CATEGORIES}
    total_valid = 0

    for r in iter_metadata_jsonl(metadata_path):
        if r.category_main in by_cat:
            img_p = PROJECT_ROOT / r.local_image_path
            if is_image_valid_and_ready(img_p):
                by_cat[r.category_main].append(r)
                total_valid += 1

    logger.info("Znaleziono łącznie %d gotowych, kompletnych próbek ze zdjęciami.", total_valid)

    rng = random.Random(seed)
    eval_sample: List[ProductMetadata] = []

    for cat in MAIN_CATEGORIES:
        pool = by_cat[cat]
        if not pool:
            logger.warning("Brak gotowych próbek dla kategorii: %s", cat)
            continue

        if sample_ratio is not None and 0.0 < sample_ratio <= 1.0:
            cnt = max(1, int(len(pool) * sample_ratio))
            sample = rng.sample(pool, cnt) if len(pool) >= cnt else pool
        elif samples_per_class and samples_per_class > 0:
            cnt = min(samples_per_class, len(pool))
            sample = rng.sample(pool, cnt) if len(pool) >= cnt else pool
        else:
            sample = pool

        eval_sample.extend(sample)
        logger.info("Kategoria %-24s: wybrano %d produktów do ewaluacji (z %d dostępnych)", cat, len(sample), len(pool))

    logger.info("Łączny rozmiar próbki ewaluacyjnej: %d produktów.", len(eval_sample))
    return eval_sample


def evaluate_existing_predictions_file(predictions_path: Path, output_dir: Path) -> None:
    """Oblicza metryki i generuje raport na podstawie gotowego pliku predykcji JSONL."""
    if not predictions_path.exists():
        raise FileNotFoundError(f"Plik predykcji nie istnieje: {predictions_path}")

    logger.info("Wczytywanie gotowych predykcji z %s...", predictions_path)
    predictions: List[CategoryPrediction] = []

    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    predictions.append(CategoryPrediction.model_validate(data))
                except Exception:
                    continue

    if not predictions:
        raise ValueError(f"Nie udało się wczytać predykcji z {predictions_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"Predykcje z {predictions_path.stem}"
    metrics = compute_metrics(predictions, model_name=model_name)

    # Macierz pomyłek
    cm_ascii = generate_confusion_matrix_ascii(predictions)
    print("\n" + cm_ascii + "\n")
    cm_png_path = output_dir / f"confusion_matrix_{predictions_path.stem}.png"
    plot_and_save_confusion_matrix(predictions, cm_png_path, model_name=model_name)

    # Analiza błędów
    err_analysis = analyze_errors_and_multilabel(predictions)
    err_path = output_dir / f"error_analysis_{predictions_path.stem}.json"
    err_path.write_text(json.dumps(err_analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    # Raport
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps([metrics.to_dict()], indent=2, ensure_ascii=False), encoding="utf-8")

    md_report_path = output_dir / "classification_report.md"
    generate_markdown_comparison_report([metrics], md_report_path)

    print("\n" + "=" * 70)
    print(f"EWALUACJA ZAKOŃCZONA SUKCESEM DLA {len(predictions)} PRODUKTÓW!")
    print(f"Top-1 Accuracy:     {metrics.top1_accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy:     {metrics.top3_accuracy * 100:.2f}%")
    print(f"Macro F1-Score:     {metrics.macro_f1 * 100:.2f}%")
    print(f"Wyniki zapisano w:  {output_dir}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark klasyfikatorów kategorii Erli (GPU/CPU)")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=RAW_DIR / "metadata.jsonl",
        help="Ścieżka do metadata.jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Wybór modelu/modeli do ewaluacji: 'all' (porównanie), 'siglip', 'clip', 'linear_probe' lub po przecinku (np. 'siglip,clip')",
    )
    parser.add_argument(
        "--single-model",
        type=str,
        default=None,
        choices=["siglip", "clip", "linear_probe"],
        help="Wskazanie dokładnie jednego konkretnego modelu do ewaluacji (np. --single-model siglip)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Wymusza ewaluację wszystkich modeli (włącznie z Linear Probe)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Liczba próbek z każdej kategorii (0 = wszystkie dostępne)",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Ułamek całego zbioru danych do włączenia do ewaluacji (np. 0.1 = 10%% dostępnych danych z bazy zamiast sztywnej liczby)",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Jeśli ustawione, ewaluuje całość dostępnych produktów ze wszystkich kategorii bez ograniczeń",
    )
    parser.add_argument(
        "--ratio",
        "--train-ratio",
        "--linear-probe-train-ratio",
        dest="train_ratio",
        type=float,
        default=0.2,
        help="Ułamek zbioru danych przeznaczony na trening Linear Probe (domyślnie: 0.2, czyli 20%% train / 80%% test)",
    )
    parser.add_argument(
        "--linear-probe-max-train-per-class",
        type=int,
        default=500,
        help="Maksymalna liczba próbek treningowych na klasę dla Linear Probe (0 = brak limitu, domyślnie: 500)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Wybór urządzenia obliczeniowego (auto / cuda / cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rozmiar paczki przy inferencji (rekomendowane: 32-64 dla GPU, 16 dla CPU)",
    )
    parser.add_argument(
        "--include-linear-probe",
        action="store_true",
        help="Czy włączyć do porównania nadzorowany Linear Probe (gdy --model all)",
    )
    parser.add_argument(
        "--evaluate-file",
        type=Path,
        default=None,
        help="Ścieżka do pliku predictions.jsonl do ewaluacji (np. z kolejki RabbitMQ)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Główny katalog na raporty",
    )
    args = parser.parse_args()

    # Określenie docelowego urządzenia
    if args.device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = args.device

    device_label = "cuda" if target_device.startswith("cuda") else "cpu"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"benchmark_{device_label}_{timestamp}"
    run_output_dir = args.output_dir / run_dir_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Wybrane urządzenie ewaluacji: %s (%s)", target_device, "GPU aktywne" if "cuda" in target_device else "CPU")
    logger.info("Katalog wyników tego uruchomienia: %s", run_output_dir)

    # Tryb 1: Ewaluacja gotowego pliku z predykcjami (np. z RabbitMQ)
    if args.evaluate_file:
        evaluate_existing_predictions_file(args.evaluate_file, run_output_dir)
        return

    # Tryb 2: Uruchomienie benchmarku modeli na reprezentatywnym podzbiorze
    effective_samples = None if (args.all_samples or args.samples_per_class <= 0) else args.samples_per_class
    eval_records = build_evaluation_sample(
        args.metadata,
        samples_per_class=effective_samples,
        sample_ratio=args.sample_ratio,
    )

    # Wybór modeli na podstawie parametrów: --single-model, --all-models, --model
    if args.single_model:
        selected_model_slugs = [args.single_model.lower().strip()]
    elif args.all_models:
        selected_model_slugs = ["siglip", "clip", "linear_probe"]
    elif args.model.lower() == "all":
        selected_model_slugs = ["siglip", "clip"]
        if args.include_linear_probe:
            selected_model_slugs.append("linear_probe")
    else:
        selected_model_slugs = [m.strip().lower() for m in args.model.split(",") if m.strip()]

    valid_models = {"siglip", "clip", "linear_probe"}
    for m in selected_model_slugs:
        if m not in valid_models:
            raise ValueError(f"Nieobsługiwany model '{m}'. Dostępne opcje to: {sorted(list(valid_models))}")

    want_linear_probe = "linear_probe" in selected_model_slugs
    want_siglip = "siglip" in selected_model_slugs
    want_clip = "clip" in selected_model_slugs

    logger.info("Wybrane modele do testu: %s", ", ".join(selected_model_slugs))

    models_to_test: List[Tuple[str, Any]] = []

    siglip_clf = None
    if want_siglip or want_linear_probe:
        siglip_clf = SigLIPClassifier(device=target_device, prompt_lang="en")
        if want_siglip:
            models_to_test.append(("SigLIP (Multi-Label Sigmoid)", siglip_clf))

    if want_clip:
        clip_clf = CLIPClassifier(device=target_device, prompt_lang="en")
        models_to_test.append(("CLIP (Softmax Baseline)", clip_clf))

    if want_linear_probe:
        logger.info("\n" + "=" * 60)
        logger.info("TRENOWANIE LINEAR PROBE NA CECHACH SIGLIP (%s)", target_device.upper())
        logger.info("=" * 60)
        rng_train = random.Random(42)

        # Podział zbalansowany na zbiór treningowy i testowy
        by_cat_eval = {c: [] for c in MAIN_CATEGORIES}
        for r in eval_records:
            if r.category_main in by_cat_eval:
                by_cat_eval[r.category_main].append(r)

        train_records = []
        train_id_set = set()
        for c in MAIN_CATEGORIES:
            pool = by_cat_eval[c]
            # Obliczenie liczby próbek treningowych na podstawie ratio i opcjonalnego limitu max
            n_target = max(5, int(len(pool) * args.train_ratio))
            if args.linear_probe_max_train_per_class and args.linear_probe_max_train_per_class > 0:
                n_target = min(n_target, args.linear_probe_max_train_per_class)
            cnt = min(n_target, len(pool) - 1 if len(pool) > 1 else len(pool))
            sampled = rng_train.sample(pool, cnt) if len(pool) >= cnt else pool
            train_records.extend(sampled)
            for s in sampled:
                train_id_set.add(s.product_id)

        # Rzetelna ewaluacja: usuwamy próbki treningowe ze zbioru testowego
        eval_records = [r for r in eval_records if r.product_id not in train_id_set]
        logger.info(
            "Wydzielono %d próbek na zbiór treningowy Linear Probe (ratio: %.1f%%, pozostało %d próbek do testu).",
            len(train_records), args.train_ratio * 100.0, len(eval_records)
        )

        if not train_records:
            logger.warning("Brak próbek treningowych dla Linear Probe, pomijam ten model.")
        else:
            train_imgs = [PROJECT_ROOT / r.local_image_path for r in train_records]
            cat2idx = {cat: i for i, cat in enumerate(MAIN_CATEGORIES)}
            import numpy as np
            targets = np.zeros((len(train_records), len(MAIN_CATEGORIES)), dtype=int)
            for i, r in enumerate(train_records):
                targets[i, cat2idx[r.category_main]] = 1

            lp_clf = LinearProbeClassifier(feature_extractor=siglip_clf)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            lp_clf.fit(train_imgs, targets, batch_size=args.batch_size)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            models_to_test.append(("Linear Probe (SigLIP Features)", lp_clf))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_images = [PROJECT_ROOT / r.local_image_path for r in eval_records]
    eval_ids = [r.product_id for r in eval_records]
    eval_true_cats = [r.category_main for r in eval_records]

    all_metrics: List[EvaluationMetrics] = []

    for model_label, clf in models_to_test:
        logger.info("\n" + "=" * 60)
        logger.info("EWALUACJA: %s na urządzeniu %s", model_label, target_device.upper())
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

        pred_path = run_output_dir / f"predictions_{slug}.jsonl"
        save_jsonl(preds, pred_path)

        # Oblicz metryki
        metrics = compute_metrics(preds, model_name=model_label)
        all_metrics.append(metrics)

        # Generuj macierz pomyłek ASCII i PNG
        cm_ascii = generate_confusion_matrix_ascii(preds)
        print("\n" + cm_ascii + "\n")
        cm_png_path = run_output_dir / f"confusion_matrix_{slug}.png"
        plot_and_save_confusion_matrix(preds, cm_png_path, model_name=model_label)

        # Analiza błędów
        err_analysis = analyze_errors_and_multilabel(preds)
        err_path = run_output_dir / f"error_analysis_{slug}.json"
        err_path.write_text(json.dumps(err_analysis, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Zakończono: %s | Top-1 Acc: %.2f%% | Top-3 Acc: %.2f%% | Macro F1: %.2f%% | Latencja: %.1f ms (%.1f FPS)",
                    model_label, metrics.top1_accuracy * 100, metrics.top3_accuracy * 100,
                    metrics.macro_f1 * 100, metrics.avg_latency_ms, metrics.throughput_fps)

    # Zapisanie całościowego raportu podsumowującego w katalogu uruchomienia
    summary_path = run_output_dir / "benchmark_summary.json"
    summary_data = [m.to_dict() for m in all_metrics]
    summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    md_report_path = run_output_dir / "classification_report.md"
    generate_markdown_comparison_report(all_metrics, md_report_path)

    # Kopiowanie najnowszych wyników również bezpośrednio do głównego katalogu reports/ dla łatwego wglądu
    try:
        shutil.copy(md_report_path, args.output_dir / "classification_report.md")
        shutil.copy(summary_path, args.output_dir / "benchmark_summary.json")
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("BENCHMARK UKOŃCZONY POMYŚLNIE!")
    print(f"Urządzenie:               {target_device.upper()}")
    print(f"Unikalny folder raportu:  {run_output_dir}")
    print(f"1. Raport Markdown:       {md_report_path}")
    print(f"2. Podsumowanie JSON:     {summary_path}")
    print(f"3. Macierze pomyłek PNG:  {run_output_dir}/confusion_matrix_*.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
