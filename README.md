# My OCR App

Self-hosted OCR API for note images. It runs a local PaddleOCR 3.7 engine with PP-OCRv6 on CPU and can also call Alibaba Cloud Model Studio (DashScope) Qwen vision models in the cloud. You pick the model per request, and the API reports per-line text, bounding boxes, and (for cloud models) token usage and estimated cost.

## Features

- Bearer-token authentication
- Multiple OCR engines: local PaddleOCR and Qwen vision models on Alibaba Cloud
- Per-request model selection with `?model=<id>` and a server-side default
- `GET /models` lists available engines with USD pricing
- JPEG, PNG, WebP, HEIC/HEIF, and TIFF input
- EXIF orientation handling and automatic transparency flattening
- Configurable image resizing and detection limits for stable CPU memory use
- Per-line confidence scores and bounding boxes; token usage and estimated cost for cloud models
- Single local OCR worker with a bounded concurrency queue (cloud models run concurrently)
- No OCR text logging
- Non-root container with a read-only filesystem

## Models

### Local PaddleOCR (default)

The runtime ships with PaddleOCR 3.7.0 and PaddlePaddle 3.2.2 on CPU.

| Role | Default model |
| --- | --- |
| Text detection | `PP-OCRv6_medium_det` |
| Text recognition | `PP-OCRv6_medium_rec` |
| Textline orientation | `PP-LCNet_x1_0_textline_ori`, only when `OCR_USE_TEXTLINE_ORIENTATION=true` |

`OCR_MODEL_TIER` selects `tiny`, `small`, or `medium`, which maps to `PP-OCRv6_<tier>_det` and `PP-OCRv6_<tier>_rec`. You can override the two models independently with `OCR_DET_MODEL_NAME` and `OCR_REC_MODEL_NAME`. Disable the local engine with `OCR_ENABLE_PADDLE=false`.

Models are downloaded on first use from HuggingFace (`PaddlePaddle/<model-name>`) by PaddleX into `/models/paddlex/official_models`, inside the `ocr-models` volume. The image sets the model source to HuggingFace and disables the source connectivity check, so model resolution does not depend on a startup network probe. Model weights are not baked into the image; for offline deployment, populate the volume in advance.

### Qwen vision models on Alibaba Cloud (optional)

Set `OCR_QWEN_API_KEY` to your Model Studio API key (Singapore region by default) to enable cloud models. At startup the app calls the provider's `/models` API and registers every model your key can access, so `/models` reflects exactly what your account offers (e.g. `qwen-vl-ocr-2025-11-20`, `qwen3.7-plus`, `qwen3.8-flash`, `qwen3-vl-plus`). To restrict the list (for example to a few you actually want to compare), set `OCR_QWEN_MODELS` to a comma-separated subset. Set `OCR_QWEN_BASE_URL` to change region/endpoint.

Images are base64-encoded into the request. The model is prompted to return a strict JSON array of lines with bounding boxes; responses are parsed tolerantly and fall back to plain text if the model does not return JSON.

Each Qwen request costs money. Cloud responses include `input_tokens`, `output_tokens`, and an `estimated_cost_usd` computed from the price catalog in `app/engines/qwen_catalog.py`. Prices change frequently — enter current USD prices per 1M tokens from the Model Studio console; unentered prices are reported as `null` and excluded from the cost estimate.

### Model selection

`OCR_DEFAULT_MODEL` sets the default engine used when `?model=` is omitted. When unset and PaddleOCR is enabled, the default is `paddle`. Otherwise the first enabled Qwen model is used. Every OCR response reports the engine actually used in the `model` field.

- Run against the local Paddle engine: `?model=paddle` (or omit `model`).
- Run against a cloud model: `?model=qwen-vl-ocr-2025-11-20`.

## Requirements

- Docker with Docker Compose
- About 2.3 GB of available memory for the container, required only when the local PaddleOCR engine is enabled (`OCR_ENABLE_PADDLE=true`)
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

Pick a specific engine (e.g. a Qwen model, or force the local Paddle engine):

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -F "file=@note.jpg" \
  "https://ocr.example.com/ocr?model=qwen-vl-ocr-2025-11-20"

curl -H "Authorization: Bearer $TOKEN" \
  -F "file=@note.jpg" \
  "https://ocr.example.com/ocr?model=paddle"
```

List the enabled engines and their USD pricing:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  https://ocr.example.com/models
```

Unknown model IDs return `400` with the list of available engines.

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
    "provider": "paddle",
    "id": "paddle",
    "label": "PaddleOCR (local)",
    "detection": "PP-OCRv6_medium_det",
    "recognition": "PP-OCRv6_medium_rec",
    "tier": "medium"
  },
  "usage": null,
  "processing_time_ms": 7240.56
}
```

Coordinates are in pixels of the processed image shown in `image.width` and `image.height`. For Qwen cloud models, `model` uses `provider: "qwen"`, `model.id` is the model ID, and `usage` reports `input_tokens`, `output_tokens`, and `estimated_cost_usd`. Cloud line `confidence` is `1.0` and not meaningful, so `min_confidence` filtering is effectively a no-op for cloud output.

Example response from a Qwen cloud model:

```json
{
  "text": "First line\nSecond line",
  "lines": [
    {
      "text": "First line",
      "confidence": 1.0,
      "box": [10.0, 10.0, 400.0, 60.0]
    },
    {
      "text": "Second line",
      "confidence": 1.0,
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
    "provider": "qwen",
    "id": "qwen-vl-ocr-2025-11-20",
    "label": "Qwen-VL OCR"
  },
  "usage": {
    "input_tokens": 1892,
    "output_tokens": 41,
    "estimated_cost_usd": null
  },
  "processing_time_ms": 4120.33
}
```

### Errors

| Status | Meaning |
| --- | --- |
| 400 | Empty or invalid image, or unknown `model` |
| 401 | Missing or invalid bearer token |
| 413 | Upload or decoded image too large |
| 415 | Unsupported content type or image format |
| 429 | Local OCR worker is busy beyond the configured timeout |
| 500 | OCR processing failed |
| 502 | OCR engine error (e.g. Qwen API rejected the request after retries) |

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
| --- | --- | --- |
| `OCR_API_KEY` | required | Bearer token, at least 16 characters |
| `OCR_ENABLE_PADDLE` | `true` | Enable the local PaddleOCR engine |
| `OCR_DEFAULT_MODEL` | derived | Engine used when `?model=` is omitted (`paddle` or a Qwen model ID) |
| `OCR_QWEN_API_KEY` | unset | Alibaba Cloud Model Studio API key; enables Qwen cloud models |
| `OCR_QWEN_BASE_URL` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | Model Studio OpenAI-compatible endpoint |
| `OCR_QWEN_MODELS` | empty | Optional comma-separated allowlist of Qwen model IDs; empty = auto-discover all models the key can access |
| `OCR_QWEN_TIMEOUT_SECONDS` | `60` | Per-request timeout to the provider |
| `OCR_QWEN_MAX_RETRIES` | `3` | Retries on 429/5xx with backoff |
| `OCR_MODEL_TIER` | `medium` | `tiny`, `small`, or `medium` |
| `OCR_DET_MODEL_NAME` | derived | Override the detection model name |
| `OCR_REC_MODEL_NAME` | derived | Override the recognition model name |
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

### TrOCR comparison

`scripts/evaluate_trocr.py` compares the single-line `microsoft/trocr-small-handwritten` recognizer with the current PaddleOCR pipeline. PaddleOCR detects text lines, then TrOCR recognizes each crop. The optional group is not installed in the runtime image.

```bash
uv sync --group trocr
uv run --group trocr python -m scripts.evaluate_trocr \
  --image example-note-sm.jpg \
  --truth text.txt \
  --device auto \
  --runs 3 \
  --save-crops /tmp/trocr-crops
```

Use `--device cpu` to compare with Apple Silicon MPS, `--preprocess contrast` to test the contrast preprocessing, and `--merge-gap-ratio 0` to disable grouping of same-line detection fragments. The model and tokenizer are downloaded from HuggingFace on first use.

Evaluation images and transcriptions are intentionally ignored by git and excluded from the Docker image.

## Privacy and Security

- Every OCR request requires a bearer token.
- Recognized text is not written to application logs.
- Images are processed in temporary files and removed after each request.
- Using a Qwen cloud model sends the (possibly resized) image to Alibaba Cloud Model Studio; the Qwen API key is only used to contact that provider.
- The container runs as a non-root user with dropped capabilities, a read-only root filesystem, and no host ports.
- Keep `.env` out of version control and readable only by its owner.
