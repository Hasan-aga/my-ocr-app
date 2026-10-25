from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.images import (
    ImageTooLargeError,
    UnsupportedImageTypeError,
    prepare_image,
)


def make_settings(**overrides) -> Settings:
    return Settings(api_key="test-key-1234567890", warmup=False, **overrides)


def encode(image: Image.Image, image_format: str = "PNG", **save_kwargs) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=image_format, **save_kwargs)
    return buffer.getvalue()


def test_flattens_transparency(tmp_path: Path) -> None:
    image = Image.new("RGBA", (100, 100), (255, 0, 0, 0))
    image.putpixel((99, 99), (255, 0, 0, 255))
    prepared = prepare_image(encode(image), make_settings())
    try:
        with Image.open(prepared.path) as result:
            assert result.mode == "RGB"
            assert result.getpixel((0, 0)) == (255, 255, 255)
            assert result.getpixel((99, 99)) == (255, 0, 0)
    finally:
        prepared.path.unlink(missing_ok=True)


def test_resizes_longest_side() -> None:
    image = Image.new("RGB", (5000, 1000), "white")
    prepared = prepare_image(encode(image), make_settings(max_side=1000))
    try:
        assert (prepared.width, prepared.height) == (1000, 200)
        assert (prepared.original_width, prepared.original_height) == (5000, 1000)
    finally:
        prepared.path.unlink(missing_ok=True)


def test_rejects_pixel_limit() -> None:
    image = Image.new("RGB", (100, 100), "white")
    with pytest.raises(ImageTooLargeError):
        prepare_image(encode(image), make_settings(max_image_pixels=5000))


def test_rejects_unsupported_format() -> None:
    image = Image.new("RGB", (10, 10), "white")
    with pytest.raises(UnsupportedImageTypeError):
        prepare_image(encode(image, "GIF"), make_settings())


def test_applies_exif_orientation() -> None:
    image = Image.new("RGB", (100, 50), "white")
    exif = Image.Exif()
    exif[274] = 6
    prepared = prepare_image(
        encode(image, "JPEG", exif=exif),
        make_settings(),
    )
    try:
        assert (prepared.original_width, prepared.original_height) == (50, 100)
    finally:
        prepared.path.unlink(missing_ok=True)


def test_contrast_preprocess_returns_rgb() -> None:
    image = Image.new("RGB", (200, 100), (230, 230, 230))
    prepared = prepare_image(encode(image), make_settings(preprocess="contrast"))
    try:
        with Image.open(prepared.path) as result:
            assert result.mode == "RGB"
            assert result.size == (200, 100)
    finally:
        prepared.path.unlink(missing_ok=True)
