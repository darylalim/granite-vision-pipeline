"""Tests for the utils module."""

import io
import time
from pathlib import Path

from pipeline.utils import temp_upload, timed


def test_timed_returns_duration() -> None:
    with timed() as t:
        time.sleep(0.01)
    assert t.duration_s >= 0.01


def test_timed_duration_is_zero_before_exit() -> None:
    with timed() as t:
        assert t.duration_s == 0.0


def test_temp_upload_creates_and_cleans_up_file() -> None:
    uploaded = io.BytesIO(b"test content")
    with temp_upload(uploaded, suffix=".pdf") as path:
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"test content"
        assert path.endswith(".pdf")
    assert not Path(path).exists()


def test_temp_upload_cleans_up_on_exception() -> None:
    uploaded = io.BytesIO(b"data")
    path_ref: str | None = None
    try:
        with temp_upload(uploaded) as path:
            path_ref = path
            raise RuntimeError("simulated error")
    except RuntimeError:
        pass
    assert path_ref is not None
    assert not Path(path_ref).exists()
