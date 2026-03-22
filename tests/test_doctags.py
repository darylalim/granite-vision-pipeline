"""Tests for the doctags module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from docling_core.types.doc.document import DoclingDocument
from PIL import Image

from pipeline.doctags import (
    create_doctags_model,
    export_markdown,
    generate_doctags,
    parse_doctags,
    render_pdf_pages,
)

TEST_PDF = str(Path(__file__).parent / "data" / "pdf" / "test_pictures.pdf")


# --- render_pdf_pages tests ---


def test_render_pdf_pages_returns_list_of_images() -> None:
    pages = render_pdf_pages(TEST_PDF)
    assert isinstance(pages, list)
    assert len(pages) > 0
    for page in pages:
        assert isinstance(page, Image.Image)


def test_render_pdf_pages_images_have_nonzero_dimensions() -> None:
    pages = render_pdf_pages(TEST_PDF)
    for page in pages:
        w, h = page.size
        assert w > 0
        assert h > 0


def test_render_pdf_pages_images_are_rgb() -> None:
    pages = render_pdf_pages(TEST_PDF)
    for page in pages:
        assert page.mode == "RGB"


# --- parse_doctags tests ---


def test_parse_doctags_returns_docling_document() -> None:
    doctags = (
        "<doctag><text><loc_50><loc_50><loc_450><loc_100>Hello world</text></doctag>"
    )
    image = Image.new("RGB", (500, 500), (255, 255, 255))
    result = parse_doctags(doctags, image)
    assert isinstance(result, DoclingDocument)


def test_parse_doctags_returns_none_for_empty_string() -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    assert parse_doctags("", image) is None


def test_parse_doctags_returns_none_for_missing_doctag_tags() -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    assert parse_doctags("just some random text", image) is None


def test_parse_doctags_handles_malformed_content() -> None:
    doctags = "<doctag>this is not valid doctags content</doctag>"
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    result = parse_doctags(doctags, image)
    assert result is None or isinstance(result, DoclingDocument)


# --- export_markdown tests ---


def test_export_markdown_returns_string() -> None:
    doc = DoclingDocument(name="test")
    result = export_markdown(doc)
    assert isinstance(result, str)


# --- create_doctags_model tests ---


@patch("pipeline.doctags.AutoModelForVision2Seq")
@patch("pipeline.doctags.AutoProcessor")
def test_create_doctags_model_loads_correct_model(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    processor, model = create_doctags_model(device="cpu")

    mock_processor_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-docling-258M"
    )
    mock_model_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-docling-258M"
    )
    assert processor is mock_processor_cls.from_pretrained.return_value
    assert model is mock_model_cls.from_pretrained.return_value.to.return_value


@patch("pipeline.doctags.AutoModelForVision2Seq")
@patch("pipeline.doctags.AutoProcessor")
def test_create_doctags_model_moves_to_device(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    create_doctags_model(device="cpu")

    mock_model_cls.from_pretrained.return_value.to.assert_called_once_with("cpu")


# --- generate_doctags tests ---


def test_generate_doctags_uses_correct_prompt() -> None:
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_processor.apply_chat_template.return_value = "formatted prompt"

    mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_processor.return_value = MagicMock()
    mock_processor.return_value.to.return_value = mock_inputs

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_processor.batch_decode.return_value = ["<doctag>content</doctag>"]

    result = generate_doctags(Image.new("RGB", (100, 100)), mock_processor, mock_model)

    call_args = mock_processor.apply_chat_template.call_args
    messages = call_args[0][0]
    text_content = [c for c in messages[0]["content"] if c["type"] == "text"]
    assert text_content[0]["text"] == "Convert this page to docling."

    assert result == "<doctag>content</doctag>"


def test_generate_doctags_returns_empty_on_empty_output() -> None:
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_processor.apply_chat_template.return_value = "prompt"
    mock_processor.return_value = MagicMock()
    mock_processor.return_value.to.return_value = {"input_ids": torch.tensor([[1, 2]])}
    mock_model.generate.return_value = torch.tensor([[1, 2]])
    mock_processor.batch_decode.return_value = [""]

    result = generate_doctags(Image.new("RGB", (10, 10)), mock_processor, mock_model)
    assert result == ""
