import os
import warnings

os.environ.setdefault("TRANSFORMERS_USE_FAST_IMAGE_PROCESSOR", "1")
warnings.filterwarnings(
    "ignore",
    message="The class `AutoModelForVision2Seq` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but not supported on MPS",
    category=UserWarning,
)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument


def create_converter() -> DocumentConverter:
    """Create a DocumentConverter with picture description enabled."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
        repo_id="ibm-granite/granite-vision-3.3-2b",
        prompt="Describe the image in three sentences. Be concise and accurate.",
        generation_config={
            "max_new_tokens": 100,
            "do_sample": True,
            "temperature": 0.2,
        },
    )
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def convert(source: str, converter: DocumentConverter | None = None) -> DoclingDocument:
    """Convert a PDF file to a DoclingDocument."""
    if converter is None:
        converter = create_converter()
    return converter.convert(source=source).document
