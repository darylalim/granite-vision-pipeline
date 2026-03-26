"""Shared model loading and generation helpers."""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def _load_vision_model(
    repo_id: str, device: str | None = None
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load an AutoProcessor and AutoModelForVision2Seq from a HuggingFace repo.

    When device is None, auto-detects: CUDA if available, else CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(repo_id)
    model = AutoModelForVision2Seq.from_pretrained(repo_id).to(device)
    return processor, model


def create_granite_vision_model(
    device: str | None = None,
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load Granite Vision 3.3 2B."""
    return _load_vision_model("ibm-granite/granite-vision-3.3-2b", device)


def generate_response(
    conversation: list[dict],
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    max_new_tokens: int = 1024,
) -> str:
    """Generate a response from a conversation using apply_chat_template.

    Handles the common pattern: tokenize conversation, generate, trim
    input tokens from output, decode. Returns decoded string, or empty
    string if the model produces no new tokens.
    """
    device = next(model.parameters()).device

    inputs = processor.apply_chat_template(  # type: ignore[operator]
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = output[:, inputs["input_ids"].shape[1] :]
    decoded = processor.decode(trimmed[0], skip_special_tokens=True)  # type: ignore[operator]
    return decoded
