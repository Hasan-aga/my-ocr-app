# My OCR App

Self-hosted OCR API for note images. It uses FastAPI and PaddleOCR PP-OCRv6 on CPU, runs behind Caddy, and returns recognized lines with confidence scores.

## Features

- Bearer-token authentication
- JPEG, PNG, WebP, HEIC/HEIF, and TIFF input
- EXIF orientation handling and automatic transparency flattening
- Configurable image resizing and detection limits for stable CPU memory use
- Per-line confidence scores and bounding boxes
- Single OCR worker with a bounded concurrency queue
- No OCR text logging and no external OCR service
- Non-root container with a read-only filesystem

## Requirements

- Docker with Docker Compose
- About 2.3 GB of available memory for the container
- A reverse proxy network named `caddy` if using the included Compose file
- `uv` and Python 3.12 only for development and tests

## Quick Start

Build the runtime image:

```bash
docker build --target runtime -t my-ocr-app:latest .
```

Create the runtime configuration:

```bash
cp .env.example .env
TOKEN=$(openssl rand -hex 32)
sed -i "s/^OCR_API_KEY=.*/OCR_API_KEY=$TOKEN/" .env
chmod 600 .env
```

Create the shared proxy network if it does not already exist:

```bash
docker network create caddy
```

Start the service:

```bash
docker compose up -d
```

The container does not publish a host port. Put it behind a reverse proxy and route to `my-ocr-app:8000`.

Example Caddy site:

```caddyfile
ocr.example.com {
	request_body {
		max_size 20MB
	}
	reverse_proxy my-ocr-app:8000
}
```

## How to Use

The health endpoint is public:

```bash
curl https://ocr.example.com/healthz
```

Send an image and authenticate with the token from `.env`:

```bash
TOKEN=$(sed -n 's/^OCR_API_KEY=//p' .env)
curl -H "Authorization: Bearer $TOKEN" \
  -F "file=@note.jpg" \
  https://ocr.example.com/ocr
```

Filter low-confidence lines:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -F "file=@note.jpg" \
  "https://ocr.example.com/ocr?min_confidence=0.7"
```

Example response:

```json
{
  "text": "First line\nSecond line",
  "lines": [
    {
      "text": "First line",
      "confidence": 0.98,
      "box": [10.0, 10.0, 400.0, 60.0]
    },
    {
      "text": "Second line",
      "confidence": 0.91,
      "box": [10.0, 70.0, 420.0, 120.0]
    }
  ],
  "line_count": 2,
  "image": {
    "width": 2000,
    "height": 1500,
    "original_width": 4032,
    "original_height": 3024,
    "format": "JPEG",
    "content_type": "image/jpeg"
  },
  "model": {
    "detection": "PP-OCRv6_medium_det",
    "recognition": "PP-OCRv6_medium_rec",
    "tier": "medium"
  },
  "processing_time_ms": 7240.56
}
```

Coordinates are in pixels of the processed image shown in `image.width` and `image.height`.

### Errors

| Status | Meaning |
| --- | --- |
| 400 | Empty or invalid image |
| 401 | Missing or invalid bearer token |
| 413 | Upload or decoded image too large |
| 415 | Unsupported content type or image format |
| 429 | OCR worker is busy beyond the configured timeout |
| 500 | OCR processing failed |

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
| --- | --- | --- |
| `OCR_API_KEY` | required | Bearer token, at least 16 characters |
| `OCR_MODEL_TIER` | `medium` | `tiny`, `small`, or `medium` |
| `OCR_PREPROCESS` | `none` | `none` or `contrast` |
| `OCR_MAX_SIDE` | `2000` | Resize longest image side |
| `OCR_DET_LIMIT_SIDE_LEN` | `960` | Detection input limit |
| `OCR_DET_LIMIT_TYPE` | `max` | Detection limit mode |
| `OCR_MAX_UPLOAD_MB` | `15` | Upload size limit |
| `OCR_MAX_IMAGE_PIXELS` | `50000000` | Decoded pixel limit |
| `OCR_CPU_THREADS` | `3` | Paddle CPU threads |
| `OCR_ENABLE_MKLDNN` | `true` | Enable oneDNN acceleration |
| `OCR_USE_TEXTLINE_ORIENTATION` | `false` | Detect upside-down text lines; adds latency |
| `OCR_WARMUP` | `true` | Run a warmup inference at startup |
| `OCR_BUSY_TIMEOUT_SECONDS` | `1.0` | Queue wait before returning 429 |
| `OCR_DOCS_ENABLED` | `false` | Expose `/docs` and OpenAPI |

The complete list is in `.env.example`.

Model weights are downloaded on first start into the `ocr-models` volume. To use a container without internet access, populate that volume with the required PaddleOCR models first.

## Development

Install dependencies and run checks:

```bash
uv sync
uv run ruff check .
uv run pytest
```

Run the integration test against a real model:

```bash
RUN_OCR_INTEGRATION=1 uv run pytest -m integration
```

Build and run the test image:

```bash
docker build --target test -t my-ocr-app:test .
docker run --rm my-ocr-app:test
```

## Evaluation

`scripts/evaluate.py` measures CER, WER, latency, and peak RSS for a local image and ground-truth transcription:

```bash
uv run python -m scripts.evaluate \
  --image example-note-sm.jpg \
  --truth text.txt \
  --runs 3
```

Evaluation images and transcriptions are intentionally ignored by git and excluded from the Docker image.

## Privacy and Security

- Every OCR request requires a bearer token.
- Recognized text is not written to application logs.
- Images are processed in temporary files and removed after each request.
- The container runs as a non-root user with dropped capabilities, a read-only root filesystem, and no host ports.
- Keep `.env` out of version control and readable only by its owner.
