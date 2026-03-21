"""Image segmentation using Granite Vision and SAM refinement."""

import re

import torch
import torch.nn.functional as F


def extract_segmentation(
    text: str,
    patch_h: int = 24,
    patch_w: int = 24,
) -> list[int] | None:
    """Parse <seg>...</seg> RLE output into a flat integer mask.

    Labels are mapped to 0 for "others" and 1 for any other label.
    Returns None if no <seg> tags found.
    """
    match = re.search(r"<seg>(.*?)</seg>", text, re.DOTALL)
    if match is None:
        return None
    rows = match.group(1).strip().split("\n")
    tokens = [token.split(" *") for row in rows for token in row.split("| ")]
    tokens = [x[0].strip() for x in tokens for _ in range(int(x[1]))]

    mask = [0 if item == "others" else 1 for item in tokens]

    total_size = patch_h * patch_w
    if len(mask) < total_size:
        mask = mask + [mask[-1]] * (total_size - len(mask))
    elif len(mask) > total_size:
        mask = mask[:total_size]
    return mask


def prepare_mask(
    mask: list[int],
    patch_h: int,
    patch_w: int,
    size: tuple[int, int],
) -> torch.Tensor:
    """Reshape flat mask to 2D, threshold to binary, interpolate to image size.

    Args:
        mask: Flat integer mask from extract_segmentation.
        patch_h: Patch grid height.
        patch_w: Patch grid width.
        size: Target (width, height) of the original image.
    """
    t = torch.as_tensor(mask).reshape((patch_h, patch_w))
    t = t.gt(0).to(dtype=torch.float32)
    t = F.interpolate(
        t[None, None],
        size=(size[1], size[0]),
        mode="nearest",
    ).squeeze()
    return t
