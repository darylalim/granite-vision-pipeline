"""DocTags generation using Granite Docling."""

import pypdfium2
import torch
from PIL import Image
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
from transformers import AutoModelForVision2Seq, AutoProcessor


def render_pdf_pages(pdf_path: str, dpi: int = 144) -> list[Image.Image]:
    """Render each page of a PDF to a PIL RGB Image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering. Default 144.
    """
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        pages: list[Image.Image] = []
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=dpi / 72)
            pil_image = bitmap.to_pil().convert("RGB")
            pages.append(pil_image)
        return pages
    finally:
        pdf.close()


def parse_doctags(doctags: str, image: Image.Image) -> DoclingDocument | None:
    """Parse raw doctags string into a DoclingDocument.

    Returns None if doctags is empty or missing <doctag> tags.
    """
    if not doctags or "<doctag>" not in doctags:
        return None
    try:
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        return DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
    except Exception:
        return None


def export_markdown(doc: DoclingDocument) -> str:
    """Export a DoclingDocument to Markdown."""
    return doc.export_to_markdown()


def create_doctags_model(
    device: str | None = None,
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load Granite Docling 258M for doctags generation.

    When device is None, auto-detects: CUDA if available, else CPU.
    MPS is excluded for consistency with other pipeline models.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "ibm-granite/granite-docling-258M"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
    return processor, model
