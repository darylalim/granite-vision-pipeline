"""Tests for the QA module."""

from PIL import Image

from pipeline.qa import resize_for_qa


# --- resize_for_qa tests ---


def test_resize_landscape_image() -> None:
    image = Image.new("RGB", (1024, 768))
    result = resize_for_qa(image)
    assert result.size == (768, 576)


def test_resize_portrait_image() -> None:
    image = Image.new("RGB", (768, 1024))
    result = resize_for_qa(image)
    assert result.size == (576, 768)


def test_small_image_unchanged() -> None:
    image = Image.new("RGB", (400, 300))
    result = resize_for_qa(image)
    assert result.size == (400, 300)


def test_exact_max_dim_unchanged() -> None:
    image = Image.new("RGB", (768, 500))
    result = resize_for_qa(image)
    assert result.size == (768, 500)


def test_resize_preserves_aspect_ratio() -> None:
    image = Image.new("RGB", (1600, 1200))
    result = resize_for_qa(image)
    w, h = result.size
    assert abs(w / h - 1600 / 1200) < 0.01


def test_resize_custom_max_dim() -> None:
    image = Image.new("RGB", (1024, 768))
    result = resize_for_qa(image, max_dim=512)
    assert result.size == (512, 384)
