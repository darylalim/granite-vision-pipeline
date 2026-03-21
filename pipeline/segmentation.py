"""Image segmentation using Granite Vision and SAM refinement."""

import re

import torch
import torch.nn.functional as F
from PIL import Image


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


def _sample_points_from_mask(
    mask: torch.Tensor,
    num_points: int,
    is_positive: bool,
) -> torch.Tensor:
    """Sample point coordinates from inside or outside the mask."""
    if num_points <= 0:
        return torch.empty((0, 2), dtype=torch.long, device=mask.device)

    m_bool = mask.bool()
    h, w = m_bool.shape
    target = m_bool if is_positive else ~m_bool

    idx_all = torch.arange(h * w, device=mask.device)
    target_indices = idx_all[target.view(-1)]

    if len(target_indices) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=mask.device)

    rand_indices = torch.randint(
        low=0, high=len(target_indices), size=(num_points,), device=mask.device
    )
    sampled = target_indices[rand_indices]

    y = sampled // w
    x = sampled % w
    return torch.stack([x, y], dim=1)


def sample_points(
    mask: torch.Tensor,
    num_pos: int = 15,
    num_neg: int = 10,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample positive and negative points from a binary mask.

    Returns (points, labels) where points are (x, y) coordinates
    and labels are 1 for positive, 0 for negative.
    When seed is None, sampling is non-deterministic.
    """
    if seed is not None:
        torch.manual_seed(seed)

    pos_coords = _sample_points_from_mask(mask, num_pos, is_positive=True)
    neg_coords = _sample_points_from_mask(mask, num_neg, is_positive=False)

    pos_labels = torch.ones(pos_coords.shape[0], dtype=torch.long, device=mask.device)
    neg_labels = torch.zeros(neg_coords.shape[0], dtype=torch.long, device=mask.device)

    points = torch.cat([pos_coords, neg_coords], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)
    return points, labels


def compute_logits_from_mask(
    mask: torch.Tensor,
    eps: float = 1e-3,
    longest_side: int = 256,
) -> torch.Tensor:
    """Convert binary mask to logits, resize and pad for SAM input.

    Returns tensor of shape (1, longest_side, longest_side).
    """
    mask = mask.to(dtype=torch.float32)
    logits = torch.logit(mask, eps=eps).unsqueeze(0).unsqueeze(0)

    h, w = mask.shape
    scale = longest_side / float(max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    logits = F.interpolate(logits, size=(new_h, new_w), mode="bilinear", antialias=True)

    pad_h = longest_side - new_h
    pad_w = longest_side - new_w
    logits = F.pad(logits, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    logits = logits.squeeze(1)

    return logits


def draw_mask(mask: Image.Image, image: Image.Image) -> Image.Image:
    """Overlay mask on image as red semi-transparent composite.

    Args:
        mask: Binary mask (mode "L", 0=background, 255=foreground).
        image: Original image.

    Returns:
        RGBA image with red overlay where mask is foreground.
    """
    # Scale mask to semi-transparent alpha (0 -> 0, 255 -> 50)
    alpha = mask.point(lambda p: 50 if p > 0 else 0)
    red_overlay = Image.new("RGBA", image.size, (255, 0, 0, 255))
    red_overlay.putalpha(alpha)
    composite = Image.alpha_composite(image.convert("RGBA"), red_overlay)
    return composite
