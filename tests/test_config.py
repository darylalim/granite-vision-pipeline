"""Tests for the pipeline config module."""

import warnings
from unittest.mock import MagicMock, patch

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from pipeline.config import convert, create_converter


def test_create_converter_returns_document_converter() -> None:
    converter = create_converter()
    assert isinstance(converter, DocumentConverter)


def test_create_converter_enables_picture_description() -> None:
    converter = create_converter()
    opts = converter.format_to_options[InputFormat.PDF].pipeline_options
    assert opts.do_picture_description is True


def test_create_converter_enables_image_generation() -> None:
    converter = create_converter()
    opts = converter.format_to_options[InputFormat.PDF].pipeline_options
    assert opts.generate_picture_images is True
    # generate_table_images is deprecated upstream; suppress warning when accessing it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        assert opts.generate_table_images is True


@patch("pipeline.config.create_converter")
def test_convert_creates_converter_when_none_provided(mock_create: MagicMock) -> None:
    mock_doc = MagicMock()
    mock_create.return_value.convert.return_value.document = mock_doc

    result = convert("test.pdf")

    mock_create.assert_called_once()
    mock_create.return_value.convert.assert_called_once_with(source="test.pdf")
    assert result is mock_doc


def test_convert_uses_provided_converter() -> None:
    mock_converter = MagicMock(spec=DocumentConverter)
    mock_doc = MagicMock()
    mock_converter.convert.return_value.document = mock_doc

    result = convert("test.pdf", converter=mock_converter)

    mock_converter.convert.assert_called_once_with(source="test.pdf")
    assert result is mock_doc
