import argparse
import json
import resource
import statistics
import sys
import time
import unicodedata
from pathlib import Path

from app.config import Settings
from app.images import prepare_image
from app.ocr import OcrEngine


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n")
    return " ".join(normalized.split())


def levenshtein(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            insert = current[column - 1] + 1
            delete = previous[column] + 1
            substitute = previous[column - 1] + (ref_char != hyp_char)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, unit: str) -> float | None:
    if unit == "char":
        units = len(reference)
    else:
        units = len(reference.split())
    if units == 0:
        return None
    return levenshtein(reference, hypothesis) / units


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("example-note-sm.jpg"))
    parser.add_argument("--truth", type=Path, default=Path("text.txt"))
    parser.add_argument("--model-tier", default="medium")
    parser.add_argument("--det-model", default=None)
    parser.add_argument("--rec-model", default=None)
    parser.add_argument("--preprocess", default="none", choices=["none", "contrast"])
    parser.add_argument("--max-side", type=int, default=2000)
    parser.add_argument("--det-limit", type=int, default=960)
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument(
        "--textline-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(
        api_key="evaluation-key-123456",
        warmup=False,
        model_tier=args.model_tier,
        det_model_name=args.det_model,
        rec_model_name=args.rec_model,
        preprocess=args.preprocess,
        max_side=args.max_side,
        det_limit_side_len=args.det_limit,
        cpu_threads=args.cpu_threads,
        use_textline_orientation=args.textline_orientation,
    )

    load_started = time.perf_counter()
    engine = OcrEngine(settings)
    load_seconds = time.perf_counter() - load_started

    image_bytes = args.image.read_bytes()
    for _ in range(args.warmup_runs):
        prepared = prepare_image(image_bytes, settings)
        try:
            engine.recognize(prepared.path)
        finally:
            prepared.path.unlink(missing_ok=True)

    timings: list[float] = []
    lines = []
    prepared = None
    for _ in range(args.runs):
        started = time.perf_counter()
        prepared = prepare_image(image_bytes, settings)
        try:
            lines = engine.recognize(prepared.path)
        finally:
            prepared.path.unlink(missing_ok=True)
        timings.append((time.perf_counter() - started) * 1000)

    predicted = "\n".join(line.text for line in lines)
    truth = args.truth.read_text(encoding="utf-8") if args.truth.exists() else ""
    normalized_predicted = normalize_text(predicted)
    normalized_truth = normalize_text(truth)
    truth_lines = [line.strip() for line in truth.splitlines() if line.strip()]
    predicted_lines = [line.text.strip() for line in lines]

    output = {
        "image": str(args.image),
        "image_size": (
            [prepared.width, prepared.height]
            if prepared is not None
            else None
        ),
        "model": engine.model_info().model_dump(),
        "preprocess": args.preprocess,
        "max_side": args.max_side,
        "det_limit_side_len": args.det_limit,
        "model_load_seconds": round(load_seconds, 3),
        "latency_ms_median": round(statistics.median(timings), 1),
        "latency_ms_runs": [round(value, 1) for value in timings],
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "line_count": len(lines),
        "mean_confidence": (
            round(statistics.mean(line.confidence for line in lines), 4)
            if lines
            else None
        ),
        "cer": error_rate(normalized_truth, normalized_predicted, "char"),
        "wer": error_rate(normalized_truth, normalized_predicted, "word"),
        "exact_line_matches": sum(
            1 for predicted_line, truth_line in zip(predicted_lines, truth_lines, strict=False)
            if predicted_line == truth_line
        ),
        "truth_line_count": len(truth_lines),
        "predicted": predicted,
        "truth": truth,
        "lines": [
            {
                "text": line.text,
                "confidence": round(line.confidence, 4),
                "box": line.box,
            }
            for line in lines
        ],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
