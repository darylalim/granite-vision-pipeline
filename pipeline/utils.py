"""Shared utility helpers for the pipeline."""

import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, BinaryIO


@contextmanager
def temp_upload(
    uploaded_file: IO[bytes] | BinaryIO, suffix: str = ".pdf"
) -> Generator[str, None, None]:
    """Write an uploaded file to a temporary path and clean up on exit."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        path = f.name
    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)


class Timer:
    """Mutable container for elapsed time, used by the `timed` context manager."""

    __slots__ = ("duration_s",)

    def __init__(self) -> None:
        self.duration_s: float = 0.0


@contextmanager
def timed() -> Generator[Timer, None, None]:
    """Measure wall-clock time in seconds.

    Usage::

        with timed() as t:
            do_work()
        print(t.duration_s)
    """
    timer = Timer()
    start = time.perf_counter_ns()
    try:
        yield timer
    finally:
        timer.duration_s = (time.perf_counter_ns() - start) / 1e9
