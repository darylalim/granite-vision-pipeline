"""Tests for the ui_helpers module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from ui_helpers import (
    _ExampleFile,
    load_example,
    show_upload_preview,
)


# --- _ExampleFile tests ---


def test_example_file_is_bytesio() -> None:
    buf = _ExampleFile(b"hello")
    assert isinstance(buf, io.BytesIO)
    assert buf.read() == b"hello"


def test_example_file_defaults() -> None:
    buf = _ExampleFile(b"data")
    assert buf.name == ""
    assert buf.size == 0


def test_example_file_init_with_args() -> None:
    buf = _ExampleFile(b"data", name="test.pdf", size=4)
    assert buf.name == "test.pdf"
    assert buf.size == 4


# --- load_example tests ---


def test_load_example_returns_example_file(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"test content")

    result = load_example(str(test_file))

    assert isinstance(result, _ExampleFile)


def test_load_example_sets_name(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.pdf"
    test_file.write_bytes(b"pdf data")

    result = load_example(str(test_file))

    assert result.name == "sample.pdf"


def test_load_example_sets_size(tmp_path: Path) -> None:
    data = b"some content here"
    test_file = tmp_path / "file.txt"
    test_file.write_bytes(data)

    result = load_example(str(test_file))

    assert result.size == len(data)


def test_load_example_contains_file_data(tmp_path: Path) -> None:
    data = b"binary content \x00\x01\x02"
    test_file = tmp_path / "data.bin"
    test_file.write_bytes(data)

    result = load_example(str(test_file))

    assert result.read() == data


def test_load_example_is_seekable(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"hello")

    result = load_example(str(test_file))
    result.read()
    result.seek(0)

    assert result.read() == b"hello"


def test_load_example_with_real_example_file() -> None:
    """Test with actual example files in the repo."""
    result = load_example("examples/sample.pdf")

    assert result.name == "sample.pdf"
    assert result.size > 0


# --- show_upload_preview tests ---


@patch("ui_helpers.st")
def test_show_upload_preview_pdf_with_size(mock_st: MagicMock) -> None:
    buf = io.BytesIO(b"fake pdf")
    buf.name = "doc.pdf"
    buf.size = 2048  # type: ignore[attr-defined]

    show_upload_preview(buf)

    mock_st.caption.assert_called_once()
    caption_arg = mock_st.caption.call_args[0][0]
    assert "doc.pdf" in caption_arg
    assert "2 KB" in caption_arg


@patch("ui_helpers.st")
def test_show_upload_preview_pdf_without_size(mock_st: MagicMock) -> None:
    buf = io.BytesIO(b"fake pdf")
    buf.name = "report.pdf"

    show_upload_preview(buf)

    mock_st.caption.assert_called_once()
    caption_arg = mock_st.caption.call_args[0][0]
    assert "report.pdf" in caption_arg


@patch("ui_helpers.st")
def test_show_upload_preview_none(mock_st: MagicMock) -> None:
    show_upload_preview(None)

    mock_st.caption.assert_not_called()
