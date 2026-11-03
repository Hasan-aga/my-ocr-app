"""Price catalog for Qwen vision models on Alibaba Cloud Model Studio.

Prices are per 1M tokens in USD for the international (ap-southeast-1)
region. They change frequently, so entries with `None` prices are shown as
"cost unknown" in `/models` and excluded from the per-request cost estimate.

Fill in the values from the Model Studio console:
https://modelstudio.console.alibabacloud.com/ap-southeast-1/model/market
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QwenCatalogEntry:
    label: str
    input_usd_per_1m: float | None
    output_usd_per_1m: float | None
    note: str | None = None


QWEN_CATALOG: dict[str, QwenCatalogEntry] = {
    "qwen-vl-ocr-2025-11-20": QwenCatalogEntry(
        label="Qwen-VL OCR",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
        note="Dedicated OCR model for documents, tables, and handwriting",
    ),
    "qwen3.8-max": QwenCatalogEntry(
        label="Qwen3.8 Max",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
        note="Flagship vision model",
    ),
    "qwen3.8-flash": QwenCatalogEntry(
        label="Qwen3.8 Flash",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
        note="Fast and cost-effective",
    ),
    "qwen3.7-max": QwenCatalogEntry(
        label="Qwen3.7 Max",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
        note="Flagship vision model",
    ),
    "qwen3.7-plus": QwenCatalogEntry(
        label="Qwen3.7 Plus",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
        note="Balanced performance, speed, and cost",
    ),
    "qwen3.7-flash": QwenCatalogEntry(
        label="Qwen3.7 Flash",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
        note="Near-flagship performance at lower cost",
    ),
    "qwen3.6-plus": QwenCatalogEntry(
        label="Qwen3.6 Plus",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
    ),
    "qwen3.6-flash": QwenCatalogEntry(
        label="Qwen3.6 Flash",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
    ),
    "qwen3-vl-plus": QwenCatalogEntry(
        label="Qwen3-VL Plus",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
        note="Legacy Qwen3-VL series",
    ),
    "qwen-vl-max": QwenCatalogEntry(
        label="Qwen-VL Max",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
        note="Legacy Qwen-VL series",
    ),
    "qwen-vl-plus": QwenCatalogEntry(
        label="Qwen-VL Plus",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
        note="Legacy Qwen-VL series",
    ),
    "qwen-vl-ocr": QwenCatalogEntry(
        label="Qwen-VL OCR",
        input_usd_per_1m=None,
        output_usd_per_1m=None,
        note="Legacy dedicated OCR model",
    ),
}


def catalog_entry(model_id: str) -> QwenCatalogEntry | None:
    return QWEN_CATALOG.get(model_id)