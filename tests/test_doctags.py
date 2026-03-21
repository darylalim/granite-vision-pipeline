"""Tests for the doctags module."""

from pathlib import Path

from docling_core.types.doc.document import DoclingDocument
from PIL import Image

from pipeline.doctags import export_markdown, parse_doctags, render_pdf_pages

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


# --- parse_doctags tests ---


def test_parse_doctags_returns_docling_document() -> None:
    doctags = (
        "<doctag><text><loc_50><loc_50><loc_450><loc_100>Hello world</text></doctag>"
    )
    image = Image.new("RGB", (500, 500), (255, 255, 255))
    result = parse_doctags(doctags, image)
    assert isinstance(result, DoclingDocument)


def test_parse_doctags_returns_none_for_empty_string() -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    assert parse_doctags("", image) is None


def test_parse_doctags_returns_none_for_missing_doctag_tags() -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    assert parse_doctags("just some random text", image) is None


def test_parse_doctags_handles_malformed_content() -> None:
    doctags = "<doctag>this is not valid doctags content</doctag>"
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    # Should either return a DoclingDocument or None, but not raise
    result = parse_doctags(doctags, image)
    assert result is None or isinstance(result, DoclingDocument)


# --- export_markdown tests ---


def test_export_markdown_returns_string() -> None:
    doc = DoclingDocument(name="test")
    result = export_markdown(doc)
    assert isinstance(result, str)
