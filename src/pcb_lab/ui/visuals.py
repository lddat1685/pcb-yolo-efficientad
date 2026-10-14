"""Display-only overlays; never modify input pixels, model scores or maps."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

CLASS_COLORS = {
    "open_circuit": "#FF595E", "short": "#FFCA3A", "mouse_bite": "#8AC926",
    "spur": "#00B5D8", "spurious_copper": "#6A4C93", "pin_hole": "#FF70AB",
}


def overlay(result, *, boxes=True, heatmap=True, opacity=.45, ground_truth=False):
    canvas = Image.fromarray(result.image.copy())
    if heatmap and result.anomaly_map is not None:
        values = result.anomaly_map.astype(np.float64)
        lo, hi = float(values.min()), float(values.max())
        scale = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)
        colors = np.stack([255 * scale, 180 * np.sqrt(scale), 255 * (1 - scale)], axis=-1).astype(np.uint8)
        canvas = Image.blend(canvas, Image.fromarray(colors), opacity)
    draw = ImageDraw.Draw(canvas)
    if ground_truth:
        for box in result.metadata.get("ground_truth_boxes", []):
            draw.rectangle(box["xyxy"], outline="white", width=1)
    if boxes:
        for box in result.detections:
            color = CLASS_COLORS.get(box["class_name"], "white")
            x1, y1, x2, y2 = box["xyxy_original"]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            text = f"{box['class_name']} {box['confidence']:.3f}"
            position = (max(0, x1), max(0, y1 - 13))
            draw.rectangle(draw.textbbox(position, text), fill="black")
            draw.text(position, text, fill=color)
    return canvas


def png_bytes(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
