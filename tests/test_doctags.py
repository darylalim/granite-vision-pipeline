"""Tests for the doctags module."""

from pathlib import Path

from PIL import Image

from pipeline.doctags import render_pdf_pages

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
