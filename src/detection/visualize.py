"""Moduł wizualizacji wyników detekcji elementów ubioru:
- Rysowanie estetycznych bounding boxów z etykietami PL i EN
- Kodowanie kolorami według grup garderoby (odzież wierzchnia, obuwie, akcesoria)
- Generowanie mozaiki wizualnej do raportu
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image, ImageColor, ImageDraw, ImageFont

from src.common.schemas import BBox, FashionDetectionResult

# Paleta kolorów dopasowana semantycznie do kategorii garderoby
COLOR_PALETTE = {
    # Sukienki / całe ciało (fiolet / magenta)
    "dress": "#9C27B0",
    "swimwear": "#E91E63",
    # Góra garderoby (błękity / lazury)
    "jacket": "#1976D2",
    "coat": "#0D47A1",
    "shirt": "#0288D1",
    "t-shirt": "#03A9F4",
    "blouse": "#00BCD4",
    "hoodie": "#3F51B5",
    "sweater": "#5C6BC0",
    # Dół garderoby (zielenie / turkusy)
    "pants": "#2E7D32",
    "jeans": "#00796B",
    "shorts": "#4CAF50",
    "skirt": "#8BC34A",
    # Obuwie (pomarańcze / brązy)
    "shoes": "#E65100",
    "boots": "#BF360C",
    "sneakers": "#F57C00",
    "high heels": "#FF5722",
    # Torby i akcesoria (złoto / bursztyn)
    "handbag": "#FBC02D",
    "backpack": "#FFA000",
    "wallet": "#FFB300",
    "watch": "#FF8F00",
    "hat": "#795548",
    "cap": "#8D6E63",
    "sunglasses": "#455A64",
    "glasses": "#607D8B",
    "scarf": "#9E9D24",
    "belt": "#5D4037",
    "tie": "#37474F",
}
DEFAULT_COLOR = "#00ACC1"


def get_label_color(label: str) -> str:
    """Zwraca kolor hex dopasowany do rodzaju ubrania."""
    return COLOR_PALETTE.get(label.lower(), DEFAULT_COLOR)


def draw_detections(
    image: Union[str, Path, Image.Image],
    detections: List[BBox],
    line_width: int = 3,
    font_size: int = 14,
) -> Image.Image:
    """Nanosi bounding boxy z etykietami na kopię przekazanego obrazu."""
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.copy().convert("RGB")

    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Skalowanie grubości linii i czcionki do wielkości obrazu
    scale = max(1.0, min(w, h) / 600.0)
    eff_line_width = max(2, int(line_width * scale))
    eff_font_size = max(12, int(font_size * scale))

    try:
        # Próba załadowania domyślnego fontu TrueType
        font = ImageFont.truetype("arial.ttf", eff_font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", eff_font_size)
        except Exception:
            font = ImageFont.load_default()

    for bbox in detections:
        color_hex = get_label_color(bbox.label)
        color = ImageColor.getrgb(color_hex)

        # 1. Rysowanie obramowania bounding boxa
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=eff_line_width)

        # 2. Przygotowanie etykiety tekstowej
        pl_name = bbox.label_pl or bbox.label
        text = f"{pl_name} ({bbox.label}) {bbox.confidence * 100:.0f}%"

        # 3. Wymiary tła etykiety
        bbox_text = draw.textbbox((x1, y1), text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        # Umieszczenie nad lub wewnątrz ramki
        bg_y1 = max(0, y1 - text_h - 6)
        bg_y2 = bg_y1 + text_h + 6
        bg_x2 = min(w, x1 + text_w + 8)

        # Tło pod tekstem dla czytelności
        draw.rectangle([x1, bg_y1, bg_x2, bg_y2], fill=color)
        draw.text((x1 + 4, bg_y1 + 2), text, fill=(255, 255, 255), font=font)

    return img


def save_detection_visualization(
    image: Union[str, Path, Image.Image],
    detections: List[BBox],
    output_path: Union[str, Path],
) -> Path:
    """Rysuje detekcje i zapisuje gotowy plik JPEG na dysk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_detections(image, detections)
    annotated.save(output_path, quality=92)
    return output_path


def create_detection_grid(
    images_with_detections: List[Tuple[Union[str, Path, Image.Image], List[BBox], str]],
    output_path: Union[str, Path],
    cols: int = 3,
    thumb_size: Tuple[int, int] = (400, 400),
) -> Path:
    """Tworzy estetyczną mozaikę (grid) przykładowych detekcji ubrań."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(images_with_detections)
    if n == 0:
        return output_path

    cols = min(cols, n)
    rows = math.ceil(n / cols)

    tw, th = thumb_size
    grid_img = Image.new("RGB", (cols * tw, rows * th), color=(245, 245, 245))

    for idx, (img_src, bboxes, title) in enumerate(images_with_detections):
        c = idx % cols
        r = idx // cols

        annotated = draw_detections(img_src, bboxes)
        annotated.thumbnail((tw, th), Image.Resampling.LANCZOS)

        # Wyśrodkowanie kafelka
        ox = c * tw + (tw - annotated.width) // 2
        oy = r * th + (th - annotated.height) // 2
        grid_img.paste(annotated, (ox, oy))

    grid_img.save(output_path, quality=90)
    return output_path
