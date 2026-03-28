"""Shared UI helper functions for Streamlit pages.

No pipeline imports — any data requiring pipeline calls is computed
at the call site and passed as a parameter.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from PIL import Image


def show_upload_preview(
    uploaded_file: object,
    total_pages: int = 0,
    selected: list[int] | None = None,
) -> None:
    """Show file info with page context as a single caption line.

    For >8 page PDFs: ``filename — size · Pages X–Y of Z``
    For ≤8 page PDFs: ``filename — size · N page(s) — All pages selected``
    """
    if uploaded_file is None:
        return

    name = getattr(uploaded_file, "name", "file")
    size = getattr(uploaded_file, "size", None)

    parts: list[str] = [f"**{name}**"]
    if size is not None:
        parts.append(f"{size / 1024:.0f} KB")
    info = " — ".join(parts)

    if total_pages > 0 and selected:
        page_start, page_end = selected[0], selected[-1]
        if total_pages > 8:
            info += f" · Pages {page_start}–{page_end} of {total_pages}"
        else:
            page_word = "page" if total_pages == 1 else "pages"
            info += f" · {total_pages} {page_word} — All pages selected"

    st.caption(info)


class _ExampleFile(io.BytesIO):
    """BytesIO wrapper with .name and .size attributes for UploadedFile compat."""

    def __init__(self, data: bytes, name: str = "", size: int = 0) -> None:
        super().__init__(data)
        self.name = name
        self.size = size


def load_example(file_path: str) -> _ExampleFile:
    """Load a file as a BytesIO wrapper that substitutes for st.UploadedFile."""
    data = Path(file_path).read_bytes()
    return _ExampleFile(data, name=Path(file_path).name, size=len(data))


def render_thumbnail_grid(
    images: list[Image.Image],
    selected_range: tuple[int, int],
    cols_per_row: int = 4,
) -> None:
    """Render page thumbnails in a grid with selected pages highlighted.

    All pages are rendered inside a container. Selected pages use
    border=True for visual highlighting; unselected use border=False.

    Args:
        images: List of PIL Images, one per page.
        selected_range: 1-based inclusive (start, end) of selected pages.
        cols_per_row: Number of columns per row.
    """
    sel_start, sel_end = selected_range
    for row_start in range(0, len(images), cols_per_row):
        row_images = images[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, (col, img) in enumerate(zip(cols, row_images)):
            page_num = row_start + i + 1  # 1-based
            is_selected = sel_start <= page_num <= sel_end
            container = col.container(border=is_selected)
            container.image(img, width="stretch")
            container.caption(f"Page {page_num}")


def clamp_page_range(start: int, end: int, max_span: int = 8) -> tuple[int, int]:
    """Clamp a page range so it spans at most max_span pages.

    If end - start + 1 exceeds max_span, narrows the range by moving end
    to start + max_span - 1. Returns (start, end) tuple.
    """
    if end - start + 1 > max_span:
        end = start + max_span - 1
    return (start, end)
