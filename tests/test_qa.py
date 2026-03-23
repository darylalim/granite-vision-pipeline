"""Tests for the QA module."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pipeline.qa import generate_qa_response, resize_for_qa


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


def test_resize_square_at_max_dim_unchanged() -> None:
    image = Image.new("RGB", (768, 768))
    result = resize_for_qa(image)
    assert result.size == (768, 768)


def test_resize_preserves_aspect_ratio() -> None:
    image = Image.new("RGB", (1600, 1200))
    result = resize_for_qa(image)
    w, h = result.size
    assert abs(w / h - 1600 / 1200) < 0.01


def test_resize_custom_max_dim() -> None:
    image = Image.new("RGB", (1024, 768))
    result = resize_for_qa(image, max_dim=512)
    assert result.size == (512, 384)


# --- generate_qa_response tests ---


def test_generate_qa_response_rejects_empty_images() -> None:
    with pytest.raises(ValueError, match="1 to 8"):
        generate_qa_response([], "What is this?", MagicMock(), MagicMock())


def test_generate_qa_response_rejects_more_than_8_images() -> None:
    images = [Image.new("RGB", (100, 100)) for _ in range(9)]
    with pytest.raises(ValueError, match="1 to 8"):
        generate_qa_response(images, "What is this?", MagicMock(), MagicMock())


@patch("pipeline.qa.generate_response")
def test_generate_qa_response_prompt_structure(
    mock_gen: MagicMock,
) -> None:
    mock_gen.return_value = "The answer is 42."

    images = [Image.new("RGB", (100, 100)) for _ in range(3)]
    result = generate_qa_response(
        images, "What is the answer?", MagicMock(), MagicMock()
    )

    conversation = mock_gen.call_args[0][0]
    content = conversation[0]["content"]

    image_entries = [c for c in content if c["type"] == "image"]
    text_entries = [c for c in content if c["type"] == "text"]
    assert len(image_entries) == 3
    assert len(text_entries) == 1
    assert text_entries[0]["text"] == "What is the answer?"

    assert mock_gen.call_args[1]["max_new_tokens"] == 1024
    assert result == "The answer is 42."


@patch("pipeline.qa.generate_response")
def test_generate_qa_response_returns_empty_on_no_new_tokens(
    mock_gen: MagicMock,
) -> None:
    mock_gen.return_value = ""
    result = generate_qa_response(
        [Image.new("RGB", (10, 10))], "question", MagicMock(), MagicMock()
    )
    assert result == ""
