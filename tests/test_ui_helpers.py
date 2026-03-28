"""Tests for the ui_helpers module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from ui_helpers import (
    _ExampleFile,
    clamp_page_range,
    load_example,
    render_thumbnail_grid,
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


# --- clamp_page_range tests ---


def test_clamp_page_range_within_limit() -> None:
    assert clamp_page_range(1, 5, max_span=8) == (1, 5)


def test_clamp_page_range_exact_limit() -> None:
    assert clamp_page_range(1, 8, max_span=8) == (1, 8)


def test_clamp_page_range_exceeds_limit() -> None:
    assert clamp_page_range(1, 12, max_span=8) == (1, 8)


def test_clamp_page_range_single_page() -> None:
    assert clamp_page_range(5, 5, max_span=8) == (5, 5)


def test_clamp_page_range_exceeds_from_middle() -> None:
    assert clamp_page_range(3, 15, max_span=8) == (3, 10)


def test_clamp_page_range_custom_max_span() -> None:
    assert clamp_page_range(1, 10, max_span=4) == (1, 4)


def test_clamp_page_range_default_max_span() -> None:
    assert clamp_page_range(1, 8) == (1, 8)
    assert clamp_page_range(1, 9) == (1, 8)


# --- render_thumbnail_grid tests ---


@patch("ui_helpers.st")
def test_render_thumbnail_grid_creates_correct_columns(mock_st: MagicMock) -> None:
    mock_cols = [MagicMock() for _ in range(4)]
    mock_st.columns.return_value = mock_cols

    images = [Image.new("RGB", (100, 100)) for _ in range(4)]
    render_thumbnail_grid(images, selected_range=(1, 2), cols_per_row=4)

    mock_st.columns.assert_called_once_with(4)


@patch("ui_helpers.st")
def test_render_thumbnail_grid_displays_all_images(mock_st: MagicMock) -> None:
    mock_cols = [MagicMock() for _ in range(3)]
    mock_st.columns.return_value = mock_cols

    images = [Image.new("RGB", (100, 100)) for _ in range(3)]
    render_thumbnail_grid(images, selected_range=(1, 3), cols_per_row=3)

    for i, col in enumerate(mock_cols):
        container = col.container.return_value
        container.image.assert_called_once_with(images[i], width="stretch")


@patch("ui_helpers.st")
def test_render_thumbnail_grid_captions_show_page_numbers(mock_st: MagicMock) -> None:
    mock_cols = [MagicMock() for _ in range(2)]
    mock_st.columns.return_value = mock_cols

    images = [Image.new("RGB", (100, 100)) for _ in range(2)]
    render_thumbnail_grid(images, selected_range=(1, 2), cols_per_row=2)

    c0 = mock_cols[0].container.return_value
    c1 = mock_cols[1].container.return_value
    c0.caption.assert_called_once_with("Page 1")
    c1.caption.assert_called_once_with("Page 2")


@patch("ui_helpers.st")
def test_render_thumbnail_grid_highlights_selected(mock_st: MagicMock) -> None:
    mock_cols = [MagicMock() for _ in range(4)]
    mock_st.columns.return_value = mock_cols

    images = [Image.new("RGB", (100, 100)) for _ in range(4)]
    render_thumbnail_grid(images, selected_range=(2, 3), cols_per_row=4)

    # Pages 2-3 (index 1-2) should use container(border=True)
    mock_cols[1].container.assert_called_once_with(border=True)
    mock_cols[2].container.assert_called_once_with(border=True)
    # Pages 1 and 4 (index 0, 3) should use container(border=False)
    mock_cols[0].container.assert_called_once_with(border=False)
    mock_cols[3].container.assert_called_once_with(border=False)


@patch("ui_helpers.st")
def test_render_thumbnail_grid_multiple_rows(mock_st: MagicMock) -> None:
    mock_row1 = [MagicMock() for _ in range(3)]
    mock_row2 = [MagicMock() for _ in range(3)]
    mock_st.columns.side_effect = [mock_row1, mock_row2]

    images = [Image.new("RGB", (100, 100)) for _ in range(5)]
    render_thumbnail_grid(images, selected_range=(1, 5), cols_per_row=3)

    assert mock_st.columns.call_count == 2
    # Third column in second row should be empty (only 5 images, not 6)
    mock_row2[2].container.assert_not_called()


@patch("ui_helpers.st")
def test_render_thumbnail_grid_empty_images(mock_st: MagicMock) -> None:
    render_thumbnail_grid([], selected_range=(1, 1), cols_per_row=4)

    mock_st.columns.assert_not_called()

