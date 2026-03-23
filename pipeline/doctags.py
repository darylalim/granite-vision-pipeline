"""DocTags generation using Granite Docling."""

import pypdfium2
import torch
from PIL import Image
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
from transformers import AutoModelForVision2Seq, AutoProcessor


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


def generate_doctags(
    image: Image.Image,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
) -> str:
    """Generate doctags from a document image.

    Infers device from the model. Returns raw doctags string,
    or empty string if model produces no output.
    """
    device = next(model.parameters()).device

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."},
            ],
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=8192)

    trimmed = output[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=False)[0].lstrip()
    return decoded
