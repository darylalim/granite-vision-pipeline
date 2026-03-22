"""Multipage QA using Granite Vision."""

from PIL import Image


def resize_for_qa(image: Image.Image, max_dim: int = 768) -> Image.Image:
    """Resize image so its longer dimension is at most max_dim pixels.

    Preserves aspect ratio using LANCZOS resampling.
    Returns the image unchanged if already within bounds.
    """
    w, h = image.size
    longer = max(w, h)
    if longer <= max_dim:
        return image
    scale = max_dim / longer
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)
