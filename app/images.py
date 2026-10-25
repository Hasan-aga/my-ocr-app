import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from app.config import Settings

register_heif_opener()
Image.MAX_IMAGE_PIXELS = None

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
    "image/tiff",
}
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "HEIF", "HEIC", "TIFF"}


class ImageValidationError(ValueError):
    status_code = 400


class UnsupportedImageTypeError(ImageValidationError):
    status_code = 415


class ImageTooLargeError(ImageValidationError):
    status_code = 413


@dataclass(frozen=True)
class PreparedImage:
    path: Path
    width: int
    height: int
    original_width: int
    original_height: int
    format: str | None


def _flatten_alpha(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def _apply_contrast(image: Image.Image) -> Image.Image:
    gray = ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=1)
    sharpened = gray.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
    return sharpened.convert("RGB")


def prepare_image(data: bytes, settings: Settings) -> PreparedImage:
    if not data:
        raise ImageValidationError("Empty image")

    try:
        with Image.open(BytesIO(data)) as source:
            image_format = (source.format or "").upper()
            if image_format not in ALLOWED_FORMATS:
                message = f"Unsupported image format: {image_format or 'unknown'}"
                raise UnsupportedImageTypeError(message)

            header_width, header_height = source.size
            if header_width * header_height > settings.max_image_pixels:
                raise ImageTooLargeError("Image pixel limit exceeded")

            source.load()
            image = ImageOps.exif_transpose(source)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Invalid image") from exc

    original_width, original_height = image.size
    if original_width * original_height > settings.max_image_pixels:
        raise ImageTooLargeError("Image pixel limit exceeded")

    image = _flatten_alpha(image)

    if max(image.size) > settings.max_side:
        image.thumbnail((settings.max_side, settings.max_side), Image.Resampling.LANCZOS)

    if settings.preprocess == "contrast":
        image = _apply_contrast(image)

    width, height = image.size
    file_descriptor, raw_path = tempfile.mkstemp(prefix="ocr-", suffix=".png", dir="/tmp")
    os.close(file_descriptor)
    path = Path(raw_path)
    try:
        image.save(path, format="PNG")
    except Exception:
        path.unlink(missing_ok=True)
        raise

    return PreparedImage(
        path=path,
        width=width,
        height=height,
        original_width=original_width,
        original_height=original_height,
        format=image_format,
    )
