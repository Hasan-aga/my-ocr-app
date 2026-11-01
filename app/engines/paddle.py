import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from app.config import Settings
from app.engines.base import EngineResult, ParsedLine
from app.schemas import ModelInfo

PADDLE_ID = "paddle"


class PaddleEngine:
    provider = "paddle"
    id = PADDLE_ID
    label: str | None = "PaddleOCR (local)"
    is_local = True

    def __init__(self, settings: Settings) -> None:
        from paddleocr import PaddleOCR

        self.settings = settings
        self.det_model = settings.resolved_det_model
        self.rec_model = settings.resolved_rec_model
        self._ocr = PaddleOCR(
            text_detection_model_name=self.det_model,
            text_recognition_model_name=self.rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=settings.use_textline_orientation,
            text_det_limit_side_len=settings.det_limit_side_len,
            text_det_limit_type=settings.det_limit_type,
            text_det_thresh=settings.det_thresh,
            text_det_box_thresh=settings.det_box_thresh,
            text_det_unclip_ratio=settings.det_unclip_ratio,
            text_rec_score_thresh=settings.rec_score_thresh,
            return_word_box=False,
            device="cpu",
            cpu_threads=settings.cpu_threads,
            enable_mkldnn=settings.enable_mkldnn,
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self.provider,
            id=self.id,
            label=self.label,
            tier=self.settings.model_tier,
            detection=self.det_model,
            recognition=self.rec_model,
        )

    def recognize(self, image_path: Path) -> EngineResult:
        results = self._ocr.predict(
            str(image_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=self.settings.use_textline_orientation,
            text_det_limit_side_len=self.settings.det_limit_side_len,
            text_det_limit_type=self.settings.det_limit_type,
            text_det_thresh=self.settings.det_thresh,
            text_det_box_thresh=self.settings.det_box_thresh,
            text_det_unclip_ratio=self.settings.det_unclip_ratio,
            text_rec_score_thresh=self.settings.rec_score_thresh,
            return_word_box=False,
        )
        if not results:
            return EngineResult(lines=[])
        return EngineResult(lines=parse_result(results[0]))

    def warmup(self) -> None:
        image = Image.new("RGB", (640, 120), "white")
        ImageDraw.Draw(image).text((20, 45), "Warmup 123", fill="black")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = Path(handle.name)
            image.save(handle, format="PNG")
        try:
            self.recognize(path)
        finally:
            path.unlink(missing_ok=True)


def _as_mapping(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        return result
    data = getattr(result, "json", None)
    if isinstance(data, Mapping):
        return data
    return {}


def _box_from_values(box: Any) -> list[float] | None:
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if len(values) != 4:
        return None
    return values


def _box_from_polygon(polygon: Any) -> list[float] | None:
    try:
        points = [(float(point[0]), float(point[1])) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_result(result: Any) -> list[ParsedLine]:
    data = _as_mapping(result)
    texts = data.get("rec_texts") or []
    scores = data.get("rec_scores") or []
    boxes = data.get("rec_boxes")
    polygons = data.get("rec_polys")

    lines: list[ParsedLine] = []
    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        box = None
        if boxes is not None and index < len(boxes):
            box = _box_from_values(boxes[index])
        if box is None and polygons is not None and index < len(polygons):
            box = _box_from_polygon(polygons[index])
        lines.append(
            ParsedLine(
                text=text,
                confidence=max(0.0, min(1.0, score)),
                box=box,
            )
        )
    return lines