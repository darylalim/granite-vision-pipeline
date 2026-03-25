"""PDF rendering utilities via pypdfium2."""

import pypdfium2
from PIL import Image


def render_pdf_pages(
    pdf_path: str,
    dpi: int = 144,
    page_indices: list[int] | None = None,
) -> list[Image.Image]:
    """Render pages of a PDF to PIL RGB Images.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering. Default 144.
        page_indices: Zero-based page indices to render. Default None renders all.
    """
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        indices = page_indices if page_indices is not None else list(range(len(pdf)))
        pages: list[Image.Image] = []
        for i in indices:
            page = pdf[i]
            bitmap = page.render(scale=dpi / 72)
            pil_image = bitmap.to_pil().convert("RGB")
            pages.append(pil_image)
        return pages
    finally:
        pdf.close()


def get_pdf_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF without rendering."""
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        return len(pdf)
    finally:
        pdf.close()
