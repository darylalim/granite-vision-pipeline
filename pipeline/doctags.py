"""DocTags generation using Granite Docling."""

import pypdfium2
from PIL import Image


def render_pdf_pages(pdf_path: str, dpi: int = 144) -> list[Image.Image]:
    """Render each page of a PDF to a PIL RGB Image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering. Default 144.
    """
    pdf = pypdfium2.PdfDocument(pdf_path)
    pages: list[Image.Image] = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=dpi / 72)
        pil_image = bitmap.to_pil().convert("RGB")
        pages.append(pil_image)
    pdf.close()
    return pages
