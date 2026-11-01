import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.images import prepare_image
from scripts.evaluate import error_rate, normalize_text, peak_rss_mb


@dataclass(frozen=True)
class DetectedLine:
    box: list[float]
    score: float


@dataclass(frozen=True)
class LineCrop:
    box: list[float]
    detection_confidence: float
    image: Image.Image


def polygon_to_box(polygon: Any) -> list[float] | None:
    try:
        points = [(float(point[0]), float(point[1])) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def union_box(boxes: list[list[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def vertical_overlap(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[3], second[3]) - max(first[1], second[1]))


def group_and_merge(lines: list[DetectedLine], merge_gap_ratio: float) -> list[DetectedLine]:
    rows: list[list[DetectedLine]] = []
    for line in sorted(lines, key=lambda item: (item.box[1], item.box[0])):
        for row in rows:
            row_box = union_box([item.box for item in row])
            min_height = min(line.box[3] - line.box[1], row_box[3] - row_box[1])
            if min_height > 0 and vertical_overlap(line.box, row_box) / min_height >= 0.5:
                row.append(line)
                break
        else:
            rows.append([line])

    merged_lines: list[DetectedLine] = []
    for row in rows:
        row.sort(key=lambda item: item.box[0])
        merged_row: list[DetectedLine] = []
        for line in row:
            if not merged_row:
                merged_row.append(line)
                continue
            previous = merged_row[-1]
            line_height = max(previous.box[3] - previous.box[1], line.box[3] - line.box[1])
            gap = line.box[0] - previous.box[2]
            if merge_gap_ratio > 0 and gap <= line_height * merge_gap_ratio:
                merged_row[-1] = DetectedLine(
                    box=union_box([previous.box, line.box]),
                    score=min(previous.score, line.score),
                )
            else:
                merged_row.append(line)
        merged_lines.extend(merged_row)

    return sorted(merged_lines, key=lambda item: (item.box[1], item.box[0]))


def result_data(result: Any) -> dict[str, Any]:
    data = getattr(result, "json", None)
    if not isinstance(data, dict):
        return {}
    nested = data.get("res")
    return nested if isinstance(nested, dict) else data


def detect_lines(
    detector: Any,
    image_path: Path,
    merge_gap_ratio: float,
    min_box_height: float,
) -> list[DetectedLine]:
    results = list(detector.predict(str(image_path), batch_size=1))
    if not results:
        return []

    data = result_data(results[0])
    polygons = data.get("dt_polys") or []
    scores = data.get("dt_scores") or []

    lines: list[DetectedLine] = []
    for index, polygon in enumerate(polygons):
        box = polygon_to_box(polygon)
        if box is None or box[3] - box[1] < min_box_height or box[2] - box[0] < min_box_height:
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        lines.append(DetectedLine(box=box, score=max(0.0, min(1.0, score))))
    return group_and_merge(lines, merge_gap_ratio)


def crop_lines(
    image_path: Path,
    lines: list[DetectedLine],
    padding: float,
) -> list[LineCrop]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    width, height = image.size
    crops: list[LineCrop] = []
    for line in lines:
        x0, y0, x1, y1 = line.box
        pad = max(4, round((y1 - y0) * padding))
        left = max(0, round(x0) - pad)
        top = max(0, round(y0) - pad)
        right = min(width, round(x1) + pad)
        bottom = min(height, round(y1) + pad)
        if right <= left or bottom <= top:
            continue
        crops.append(
            LineCrop(
                box=line.box,
                detection_confidence=line.score,
                image=image.crop((left, top, right, bottom)),
            )
        )
    return crops


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")
    return requested


def create_detector(settings: Settings) -> Any:
    from paddleocr import TextDetection

    return TextDetection(
        model_name=settings.resolved_det_model,
        limit_side_len=settings.det_limit_side_len,
        limit_type=settings.det_limit_type,
        thresh=settings.det_thresh,
        box_thresh=settings.det_box_thresh,
        unclip_ratio=settings.det_unclip_ratio,
        device="cpu",
        cpu_threads=settings.cpu_threads,
        enable_mkldnn=settings.enable_mkldnn,
    )


def load_trocr(model_id: str, device: str, cpu_threads: int) -> tuple[Any, Any]:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    if device == "cpu":
        torch.set_num_threads(cpu_threads)
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    if getattr(model.generation_config, "max_length", None) is not None:
        model.generation_config.max_length = None
    return processor, model


def token_confidences(model: Any, outputs: Any, eos_token_id: int | None) -> list[float]:
    import torch

    scores = getattr(outputs, "scores", None)
    if not scores:
        return [0.0 for _ in range(outputs.sequences.shape[0])]

    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        scores,
        beam_indices=getattr(outputs, "beam_indices", None),
        normalize_logits=True,
    )
    confidences: list[float] = []
    for index in range(transition_scores.shape[0]):
        sequence = outputs.sequences[index, 1:]
        end = sequence.shape[0]
        if eos_token_id is not None:
            eos_positions = (sequence == eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel():
                end = int(eos_positions[0].item())
        line_scores = transition_scores[index, :end]
        line_scores = line_scores[torch.isfinite(line_scores)]
        confidences.append(
            float(line_scores.exp().mean().item()) if line_scores.numel() else 0.0
        )
    return confidences


def run_trocr(
    processor: Any,
    model: Any,
    crops: list[LineCrop],
    device: str,
    max_new_tokens: int,
    num_beams: int,
    batch_size: int,
) -> list[tuple[str, float]]:
    import torch

    results: list[tuple[str, float]] = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        pixel_values = processor(
            images=[crop.image for crop in batch],
            return_tensors="pt",
        ).pixel_values.to(device)
        with torch.inference_mode():
            outputs = model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
            )
        texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        confidences = token_confidences(model, outputs, processor.tokenizer.eos_token_id)
        results.extend(
            (text.strip(), confidence)
            for text, confidence in zip(texts, confidences, strict=False)
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("example-note-sm.jpg"))
    parser.add_argument("--truth", type=Path, default=Path("text.txt"))
    parser.add_argument("--model-id", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--model-tier", choices=["tiny", "small", "medium"], default="medium")
    parser.add_argument("--det-model", default=None)
    parser.add_argument("--preprocess", choices=["none", "contrast"], default="none")
    parser.add_argument("--max-side", type=int, default=2000)
    parser.add_argument("--det-limit", type=int, default=960)
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--merge-gap-ratio", type=float, default=1.0)
    parser.add_argument("--min-box-height", type=float, default=6.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--save-crops", type=Path, default=None)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must not be negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.num_beams < 1:
        parser.error("--num-beams must be at least 1")
    if args.padding < 0:
        parser.error("--padding must not be negative")
    if args.merge_gap_ratio < 0:
        parser.error("--merge-gap-ratio must not be negative")
    return args


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = parse_args()
    settings = Settings(
        api_key="evaluation-key-123456",
        warmup=False,
        model_tier=args.model_tier,
        det_model_name=args.det_model,
        preprocess=args.preprocess,
        max_side=args.max_side,
        det_limit_side_len=args.det_limit,
        cpu_threads=args.cpu_threads,
    )

    image_bytes = args.image.read_bytes()
    prepared = prepare_image(image_bytes, settings)
    try:
        detector_started = time.perf_counter()
        detector = create_detector(settings)
        detector_load_seconds = time.perf_counter() - detector_started

        detection_started = time.perf_counter()
        lines = detect_lines(
            detector,
            prepared.path,
            args.merge_gap_ratio,
            args.min_box_height,
        )
        detection_seconds = time.perf_counter() - detection_started
        if not lines:
            raise SystemExit("No text lines detected")

        crops = crop_lines(prepared.path, lines, args.padding)
        if not crops:
            raise SystemExit("No text line crops produced")

        crop_paths: list[str | None] = [None for _ in crops]
        if args.save_crops is not None:
            args.save_crops.mkdir(parents=True, exist_ok=True)
            for index, crop in enumerate(crops, start=1):
                crop_path = args.save_crops / f"line-{index:02d}.png"
                crop.image.save(crop_path, format="PNG")
                crop_paths[index - 1] = str(crop_path)

        device = resolve_device(args.device)
        trocr_started = time.perf_counter()
        processor, model = load_trocr(args.model_id, device, args.cpu_threads)
        trocr_load_seconds = time.perf_counter() - trocr_started

        for _ in range(args.warmup_runs):
            run_trocr(
                processor,
                model,
                crops,
                device,
                args.max_new_tokens,
                args.num_beams,
                args.batch_size,
            )

        timings: list[float] = []
        results: list[tuple[str, float]] = []
        for _ in range(args.runs):
            recognition_started = time.perf_counter()
            results = run_trocr(
                processor,
                model,
                crops,
                device,
                args.max_new_tokens,
                args.num_beams,
                args.batch_size,
            )
            timings.append((time.perf_counter() - recognition_started) * 1000)

        truth = args.truth.read_text(encoding="utf-8") if args.truth.exists() else ""
        predicted = "\n".join(text for text, _ in results)
        normalized_predicted = normalize_text(predicted)
        normalized_truth = normalize_text(truth)
        truth_lines = [line.strip() for line in truth.splitlines() if line.strip()]
        predicted_lines = [text for text, _ in results]

        output = {
            "image": str(args.image),
            "image_size": [prepared.width, prepared.height],
            "detection_model": settings.resolved_det_model,
            "recognition_model": args.model_id,
            "device": device,
            "preprocess": args.preprocess,
            "max_side": args.max_side,
            "det_limit_side_len": args.det_limit,
            "detector_load_seconds": round(detector_load_seconds, 3),
            "trocr_load_seconds": round(trocr_load_seconds, 3),
            "detection_ms": round(detection_seconds * 1000, 1),
            "recognition_ms_runs": [round(value, 1) for value in timings],
            "recognition_ms_median": round(statistics.median(timings), 1),
            "processing_ms_median": round(
                detection_seconds * 1000 + statistics.median(timings),
                1,
            ),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "line_count": len(results),
            "mean_confidence": (
                round(statistics.mean(confidence for _, confidence in results), 4)
                if results
                else None
            ),
            "cer": error_rate(normalized_truth, normalized_predicted, "char"),
            "wer": error_rate(normalized_truth, normalized_predicted, "word"),
            "exact_line_matches": sum(
                1
                for predicted_line, truth_line in zip(
                    predicted_lines,
                    truth_lines,
                    strict=False,
                )
                if predicted_line == truth_line
            ),
            "truth_line_count": len(truth_lines),
            "predicted": predicted,
            "truth": truth,
            "lines": [
                {
                    "text": text,
                    "confidence": round(confidence, 4),
                    "detection_confidence": round(crop.detection_confidence, 4),
                    "box": crop.box,
                    "crop": crop_path,
                }
                for (text, confidence), crop, crop_path in zip(
                    results,
                    crops,
                    crop_paths,
                    strict=False,
                )
            ],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    finally:
        prepared.path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
