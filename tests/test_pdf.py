"""Tests for the PDF rendering module."""

from pathlib import Path

from PIL import Image

from pipeline.pdf import get_pdf_page_count, render_pdf_pages

TEST_PDF = str(Path(__file__).parent / "data" / "pdf" / "test_pictures.pdf")


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


def test_render_pdf_pages_with_page_indices() -> None:
    all_pages = render_pdf_pages(TEST_PDF)
    first_page = render_pdf_pages(TEST_PDF, page_indices=[0])
    assert len(first_page) == 1
    assert first_page[0].size == all_pages[0].size


def test_get_pdf_page_count_returns_correct_count() -> None:
    all_pages = render_pdf_pages(TEST_PDF)
    count = get_pdf_page_count(TEST_PDF)
    assert count == len(all_pages)


def test_get_pdf_page_count_positive() -> None:
    assert get_pdf_page_count(TEST_PDF) > 0
