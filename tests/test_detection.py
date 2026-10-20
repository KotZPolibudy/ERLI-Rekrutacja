"""Testy jednostkowe dla modułu detekcji elementów ubioru (src.detection)."""

from pathlib import Path
from PIL import Image

from src.common.schemas import BBox, FashionDetectionResult
from src.detection.detector import FASHION_CLASSES_EN, FASHION_LABELS_PL
from src.detection.visualize import draw_detections, get_label_color


def test_bbox_schema_and_properties():
    """Weryfikuje poprawność schematu BBox i obliczania wymiarów oraz powierzchni."""
    box = BBox(
        x1=10.0,
        y1=20.0,
        x2=110.0,
        y2=220.0,
        label="dress",
        label_pl="sukienka",
        confidence=0.88,
    )
    assert box.width == 100.0
    assert box.height == 200.0
    assert box.area == 20000.0
    assert box.label == "dress"
    assert box.label_pl == "sukienka"
    assert box.confidence == 0.88


def test_fashion_detection_result_schema():
    """Weryfikuje strukturę wyniku detekcji z listą bounding boxów."""
    box1 = BBox(x1=0, y1=0, x2=50, y2=50, label="shoes", confidence=0.9)
    box2 = BBox(x1=50, y1=50, x2=150, y2=250, label="dress", confidence=0.95)

    res = FashionDetectionResult(
        product_id="prod_123",
        image_path="data/raw/Moda/test.jpg",
        image_width=200,
        image_height=300,
        detections=[box1, box2],
        item_count=2,
        dominant_item="dress",
        latency_ms=12.5,
    )

    assert res.product_id == "prod_123"
    assert res.item_count == 2
    assert res.dominant_item == "dress"
    assert len(res.detections) == 2


def test_fashion_dictionary_coverage():
    """Sprawdza czy wszystkie klasy modowe mają zdefiniowane polskie odpowiedniki."""
    assert len(FASHION_CLASSES_EN) >= 20
    for cls_en in FASHION_CLASSES_EN:
        assert cls_en in FASHION_LABELS_PL
        assert len(FASHION_LABELS_PL[cls_en]) > 0


def test_draw_detections_on_image(tmp_path: Path):
    """Sprawdza poprawność wizualizacji i nanoszenia ramek na obraz PIL."""
    test_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    bboxes = [
        BBox(x1=20, y1=20, x2=80, y2=160, label="dress", label_pl="sukienka", confidence=0.85),
        BBox(x1=90, y1=120, x2=140, y2=170, label="shoes", label_pl="buty", confidence=0.92),
    ]

    annotated = draw_detections(test_img, bboxes)
    assert annotated.size == (200, 200)

    # Zapis i ponowny odczyt
    save_path = tmp_path / "annotated.jpg"
    annotated.save(save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_label_color_palette():
    """Weryfikuje poprawność pobierania kolorów dla różnych typów odzieży."""
    c_dress = get_label_color("dress")
    c_shoes = get_label_color("shoes")
    c_unknown = get_label_color("unknown_xyz")

    assert c_dress.startswith("#")
    assert c_shoes.startswith("#")
    assert c_dress != c_shoes
    assert c_unknown.startswith("#")
