"""Moduł detekcji elementów ubioru dla kategorii modowej w marketplace Erli.
Wykorzystuje open-vocabulary model YOLO-World (ultralytics) zoptymalizowany pod GPU CUDA.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

# Zapewnienie kodowania UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image
import torch
from ultralytics import YOLOWorld

from src.common.schemas import BBox, FashionDetectionResult

logger = logging.getLogger("fashion_detector")

# Kompletny słownik klas modowych dostosowany do katalogu Erli
FASHION_CLASSES_EN: List[str] = [
    "dress",
    "jacket",
    "coat",
    "shirt",
    "t-shirt",
    "blouse",
    "hoodie",
    "sweater",
    "pants",
    "jeans",
    "shorts",
    "skirt",
    "shoes",
    "boots",
    "sneakers",
    "high heels",
    "handbag",
    "backpack",
    "wallet",
    "watch",
    "hat",
    "cap",
    "sunglasses",
    "glasses",
    "scarf",
    "belt",
    "tie",
    "swimwear",
]

FASHION_LABELS_PL: Dict[str, str] = {
    "dress": "sukienka",
    "jacket": "kurtka / marynarka",
    "coat": "płaszcz",
    "shirt": "koszula",
    "t-shirt": "koszulka / t-shirt",
    "blouse": "bluzka",
    "hoodie": "bluza z kapturem",
    "sweater": "sweter",
    "pants": "spodnie",
    "jeans": "dżinsy",
    "shorts": "szorty",
    "skirt": "spódnica",
    "shoes": "buty",
    "boots": "botki / kozaki",
    "sneakers": "sneakersy",
    "high heels": "szpilki",
    "handbag": "torebka",
    "backpack": "plecak",
    "wallet": "portfel",
    "watch": "zegarek",
    "hat": "czapka / kapelusz",
    "cap": "czapka z daszkiem",
    "sunglasses": "okulary przeciwsłoneczne",
    "glasses": "okulary",
    "scarf": "szalik / apaszka",
    "belt": "pasek",
    "tie": "krawat",
    "swimwear": "strój kąpielowy",
}


class FashionDetector:
    """Wysokowydajny detektor elementów ubioru zoptymalizowany dla e-commerce i GPU."""

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        device: Optional[str] = None,
        classes: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = classes or FASHION_CLASSES_EN

        logger.info("Ładowanie modelu detekcji %s na urządzenie %s...", self.model_name, self.device)
        self.model = YOLOWorld(self.model_name)
        # Ustawienie słownika klas modowych w modelu open-vocabulary
        self.model.set_classes(self.target_classes)
        logger.info("Zainicjalizowano FashionDetector z %d klasami garderoby.", len(self.target_classes))

    def detect_image(
        self,
        image_input: Union[str, Path, Image.Image],
        product_id: str = "unknown",
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> FashionDetectionResult:
        """Wykrywa elementy ubioru na pojedynczym obrazie."""
        batch_res = self.detect_batch(
            images=[image_input],
            product_ids=[product_id],
            conf=conf,
            iou=iou,
            batch_size=1,
        )
        return batch_res[0]

    def detect_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        product_ids: Optional[List[str]] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        batch_size: int = 32,
    ) -> List[FashionDetectionResult]:
        """Przetwarza wsadowo zbiór zdjęć modowych z pamięciowo bezpiecznym podziałem na paczki."""
        conf_val = conf if conf is not None else self.conf_threshold
        iou_val = iou if iou is not None else self.iou_threshold
        results: List[FashionDetectionResult] = []
        n_total = len(images)

        for start_idx in range(0, n_total, batch_size):
            end_idx = min(start_idx + batch_size, n_total)
            chunk_imgs = images[start_idx:end_idx]
            chunk_pids = product_ids[start_idx:end_idx] if product_ids else None

            # Przygotowanie wejść
            inputs = []
            paths_str = []
            img_sizes = []

            for img in chunk_imgs:
                if isinstance(img, (str, Path)):
                    p = Path(img)
                    paths_str.append(str(p))
                    inputs.append(str(p))
                    try:
                        with Image.open(p) as pil_im:
                            img_sizes.append(pil_im.size)
                    except Exception:
                        img_sizes.append((None, None))
                elif isinstance(img, Image.Image):
                    paths_str.append("in_memory")
                    inputs.append(img.convert("RGB"))
                    img_sizes.append(img.size)
                else:
                    paths_str.append("unknown")
                    inputs.append(img)
                    img_sizes.append((None, None))

            t0 = time.perf_counter()
            # Predykcja wsadowa na GPU/CPU
            preds = self.model.predict(
                inputs,
                device=self.device,
                conf=conf_val,
                iou=iou_val,
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_item_ms = elapsed_ms / max(1, len(chunk_imgs))

            for i, pred in enumerate(preds):
                p_id = chunk_pids[i] if chunk_pids else f"item_{start_idx + i}"
                path_s = paths_str[i]
                w, h = img_sizes[i]

                bboxes: List[BBox] = []
                if pred.boxes is not None and len(pred.boxes) > 0:
                    boxes_data = pred.boxes.data.cpu().numpy()
                    for b in boxes_data:
                        x1, y1, x2, y2, score, cls_id = b[:6]
                        cls_idx = int(cls_id)
                        label_en = self.target_classes[cls_idx] if cls_idx < len(self.target_classes) else f"class_{cls_idx}"
                        label_pl = FASHION_LABELS_PL.get(label_en, label_en)

                        bboxes.append(BBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            label=label_en,
                            label_pl=label_pl,
                            confidence=float(score),
                        ))

                # Wybór elementu dominującego (największy area z dobrą pewnością)
                dominant = None
                if bboxes:
                    dominant = max(bboxes, key=lambda b: b.area * b.confidence).label

                results.append(FashionDetectionResult(
                    product_id=p_id,
                    image_path=path_s,
                    image_width=w,
                    image_height=h,
                    detections=bboxes,
                    item_count=len(bboxes),
                    dominant_item=dominant,
                    latency_ms=per_item_ms,
                ))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results
