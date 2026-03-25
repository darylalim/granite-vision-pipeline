"""Tests for the ui_helpers module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from PIL import Image

from ui_helpers import (
    _ExampleFile,
    load_example,
    show_help,
    show_metrics_bar,
    show_sidebar_status,
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


# --- show_metrics_bar tests ---


@patch("ui_helpers.st")
def test_show_metrics_bar_creates_columns(mock_st: MagicMock) -> None:
    mock_cols = [MagicMock(), MagicMock()]
    mock_st.columns.return_value = mock_cols

    show_metrics_bar({"A": 1, "B": 2})

    mock_st.columns.assert_called_once_with(2)
    mock_cols[0].metric.assert_called_once_with("A", 1)
    mock_cols[1].metric.assert_called_once_with("B", 2)


@patch("ui_helpers.st")
def test_show_metrics_bar_empty_dict(mock_st: MagicMock) -> None:
    show_metrics_bar({})

    mock_st.columns.assert_not_called()


@patch("ui_helpers.st")
def test_show_metrics_bar_single_metric(mock_st: MagicMock) -> None:
    mock_col = MagicMock()
    mock_st.columns.return_value = [mock_col]

    show_metrics_bar({"Duration (s)": "1.23"})

    mock_st.columns.assert_called_once_with(1)
    mock_col.metric.assert_called_once_with("Duration (s)", "1.23")


# --- show_sidebar_status tests ---


@patch("ui_helpers.st")
def test_show_sidebar_status_shows_models(mock_st: MagicMock) -> None:
    show_sidebar_status({"Granite Vision": True, "Docling": False})

    mock_st.markdown.assert_any_call("**Models**")
    text_calls = mock_st.text.call_args_list
    assert call("Granite Vision: Loaded") in text_calls
    assert call("Docling: Not loaded") in text_calls


# --- show_upload_preview tests ---


@patch("ui_helpers.st")
@patch("ui_helpers.Image")
def test_show_upload_preview_single_image(
    mock_image: MagicMock, mock_st: MagicMock
) -> None:
    mock_st.columns.return_value = [MagicMock()]
    mock_image.open.return_value = MagicMock()

    buf = io.BytesIO(b"fake image data")
    buf.name = "photo.png"  # type: ignore[attr-defined]

    show_upload_preview(buf)

    mock_st.columns.assert_called_once_with(1)
    mock_st.image.assert_called_once()


@patch("ui_helpers.st")
def test_show_upload_preview_pdf(mock_st: MagicMock) -> None:
    mock_st.columns.return_value = [MagicMock()]

    buf = io.BytesIO(b"fake pdf")
    buf.name = "doc.pdf"  # type: ignore[attr-defined]
    buf.size = 2048  # type: ignore[attr-defined]

    show_upload_preview(buf)

    mock_st.caption.assert_called_once()
    caption_arg = mock_st.caption.call_args[0][0]
    assert "doc.pdf" in caption_arg
    assert "2 KB" in caption_arg


@patch("ui_helpers.st")
def test_show_upload_preview_multiple_files(mock_st: MagicMock) -> None:
    mock_cols = [MagicMock(), MagicMock()]
    mock_st.columns.return_value = mock_cols

    files = []
    for name in ["a.png", "b.png"]:
        img = Image.new("RGB", (5, 5))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        buf.name = name  # type: ignore[attr-defined]
        files.append(buf)

    show_upload_preview(files)

    mock_st.columns.assert_called_once_with(2)


@patch("ui_helpers.st")
def test_show_upload_preview_none_filtered(mock_st: MagicMock) -> None:
    show_upload_preview([None, None])

    mock_st.columns.assert_not_called()


@patch("ui_helpers.st")
def test_show_upload_preview_empty_list(mock_st: MagicMock) -> None:
    show_upload_preview([])

    mock_st.columns.assert_not_called()


# --- show_help tests ---


@patch("ui_helpers.st")
def test_show_help_renders_all_sections(mock_st: MagicMock) -> None:
    show_help(
        supported_formats="PDF, PNG",
        description="Upload a file to process.",
        model_info="granite-vision-3.3-2b",
    )

    mock_st.expander.assert_called_once_with("How this works")
    markdown_calls = [c[0][0] for c in mock_st.markdown.call_args_list]
    assert any("PDF, PNG" in c for c in markdown_calls)
    assert any("Upload a file" in c for c in markdown_calls)
    assert any("granite-vision" in c for c in markdown_calls)


@patch("ui_helpers.st")
def test_show_help_calls_markdown_three_times(mock_st: MagicMock) -> None:
    show_help(
        supported_formats="JPG",
        description="Desc.",
        model_info="Model.",
    )

    assert mock_st.markdown.call_count == 3
