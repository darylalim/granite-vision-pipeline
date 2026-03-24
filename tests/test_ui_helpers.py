"""Tests for the ui_helpers module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from PIL import Image

from ui_helpers import _ExampleFile, load_example, show_metrics_bar, show_sidebar_status


# --- _ExampleFile tests ---


def test_example_file_is_bytesio() -> None:
    buf = _ExampleFile(b"hello")
    assert isinstance(buf, io.BytesIO)
    assert buf.read() == b"hello"


def test_example_file_supports_name_and_size() -> None:
    buf = _ExampleFile(b"data")
    buf.name = "test.pdf"
    buf.size = 4
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
    result = load_example("examples/sample.jpg")

    assert result.name == "sample.jpg"
    assert result.size > 0
    # Verify it's a valid image
    img = Image.open(result)
    assert img.size == (640, 480)


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
    show_sidebar_status({"Granite Vision": True, "SAM": False})

    # with st.sidebar: sets context, but calls go to st.markdown/st.text
    mock_st.markdown.assert_any_call("**Models**")
    text_calls = mock_st.text.call_args_list
    assert call("Granite Vision: Loaded") in text_calls
    assert call("SAM: Not loaded") in text_calls


@patch("ui_helpers.st")
def test_show_sidebar_status_shows_index_count(mock_st: MagicMock) -> None:
    show_sidebar_status({"Model": True}, index_count=5)

    mock_st.markdown.assert_any_call("**Search Index**")
    mock_st.text.assert_any_call("5 documents indexed")


@patch("ui_helpers.st")
def test_show_sidebar_status_singular_document(mock_st: MagicMock) -> None:
    show_sidebar_status({"Model": True}, index_count=1)

    mock_st.text.assert_any_call("1 document indexed")


@patch("ui_helpers.st")
def test_show_sidebar_status_no_index_count(mock_st: MagicMock) -> None:
    show_sidebar_status({"Model": True})

    # Should show Models but not Search Index
    markdown_calls = [c[0][0] for c in mock_st.markdown.call_args_list]
    assert "**Models**" in markdown_calls
    assert "**Search Index**" not in markdown_calls
