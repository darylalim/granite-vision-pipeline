"""Shared model loading and generation helpers."""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def _load_vision_model(
    repo_id: str, device: str | None = None
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load an AutoProcessor and AutoModelForVision2Seq from a HuggingFace repo.

    When device is None, auto-detects: CUDA if available, else CPU.
    MPS is excluded due to limited operator support in SAM/transformers.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(repo_id)
    model = AutoModelForVision2Seq.from_pretrained(repo_id).to(device)
    return processor, model


def create_granite_vision_model(
    device: str | None = None,
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load Granite Vision 3.3 2B.

    Shared across segmentation, QA, and RAG answer generation.
    """
    return _load_vision_model("ibm-granite/granite-vision-3.3-2b", device)


def create_doctags_model(
    device: str | None = None,
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load Granite Docling 258M for doctags generation."""
    return _load_vision_model("ibm-granite/granite-docling-258M", device)
