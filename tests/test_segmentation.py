"""Tests for the segmentation module."""

import torch
from PIL import Image

from pipeline.segmentation import (
    compute_logits_from_mask,
    draw_mask,
    extract_segmentation,
    prepare_mask,
    sample_points,
)


# --- extract_segmentation tests ---


def test_extract_segmentation_valid_text() -> None:
    text = "<seg>others *3\n dog *1| others *2\n others *3</seg>"
    result = extract_segmentation(text, patch_h=3, patch_w=3)
    assert result is not None
    assert len(result) == 9
    # Row 0: others others others -> 0 0 0
    # Row 1: dog others others -> 1 0 0
    # Row 2: others others others -> 0 0 0
    assert result == [0, 0, 0, 1, 0, 0, 0, 0, 0]


def test_extract_segmentation_no_seg_tags() -> None:
    result = extract_segmentation("no tags here")
    assert result is None


def test_extract_segmentation_pads_short_mask() -> None:
    text = "<seg>others *2</seg>"
    result = extract_segmentation(text, patch_h=2, patch_w=2)
    assert result is not None
    assert len(result) == 4
    # Pads with last value (0 for "others")
    assert result == [0, 0, 0, 0]


def test_extract_segmentation_truncates_long_mask() -> None:
    text = "<seg>cat *6</seg>"
    result = extract_segmentation(text, patch_h=2, patch_w=2)
    assert result is not None
    assert len(result) == 4
    assert result == [1, 1, 1, 1]


def test_extract_segmentation_multiple_labels() -> None:
    text = "<seg>others *1| cat *1| dog *1</seg>"
    result = extract_segmentation(text, patch_h=1, patch_w=3)
    assert result is not None
    assert result == [0, 1, 1]


# --- prepare_mask tests ---


def test_prepare_mask_shape() -> None:
    mask = [0, 1, 1, 0]
    result = prepare_mask(mask, patch_h=2, patch_w=2, size=(100, 80))
    assert result.shape == (80, 100)


def test_prepare_mask_binary_values() -> None:
    mask = [0, 1, 1, 0]
    result = prepare_mask(mask, patch_h=2, patch_w=2, size=(10, 10))
    unique = torch.unique(result)
    assert all(v in (0.0, 1.0) for v in unique.tolist())


def test_prepare_mask_thresholding() -> None:
    # All zeros -> all 0.0, all ones -> all 1.0
    result_zero = prepare_mask([0, 0, 0, 0], patch_h=2, patch_w=2, size=(4, 4))
    assert result_zero.sum().item() == 0.0

    result_one = prepare_mask([1, 1, 1, 1], patch_h=2, patch_w=2, size=(4, 4))
    assert result_one.sum().item() == 16.0


# --- sample_points tests ---


def test_sample_points_counts() -> None:
    mask = torch.zeros(10, 10)
    mask[:5, :5] = 1.0
    points, labels = sample_points(mask, num_pos=5, num_neg=3, seed=42)
    assert points.shape == (8, 2)
    assert labels.shape == (8,)
    assert (labels == 1).sum().item() == 5
    assert (labels == 0).sum().item() == 3


def test_sample_points_within_bounds() -> None:
    mask = torch.zeros(20, 30)
    mask[5:15, 10:20] = 1.0
    points, labels = sample_points(mask, num_pos=10, num_neg=5, seed=42)
    assert (points[:, 0] >= 0).all() and (points[:, 0] < 30).all()  # x < width
    assert (points[:, 1] >= 0).all() and (points[:, 1] < 20).all()  # y < height


def test_sample_points_deterministic_with_seed() -> None:
    mask = torch.zeros(10, 10)
    mask[:5, :] = 1.0
    p1, l1 = sample_points(mask, seed=123)
    p2, l2 = sample_points(mask, seed=123)
    assert torch.equal(p1, p2)
    assert torch.equal(l1, l2)


def test_sample_points_all_zero_mask() -> None:
    mask = torch.zeros(10, 10)
    points, labels = sample_points(mask, num_pos=5, num_neg=3, seed=42)
    # No foreground -> 0 positive points, 3 negative points
    assert (labels == 1).sum().item() == 0
    assert (labels == 0).sum().item() == 3


def test_sample_points_all_one_mask() -> None:
    mask = torch.ones(10, 10)
    points, labels = sample_points(mask, num_pos=5, num_neg=3, seed=42)
    # No background -> 5 positive points, 0 negative points
    assert (labels == 1).sum().item() == 5
    assert (labels == 0).sum().item() == 0


# --- compute_logits_from_mask tests ---


def test_compute_logits_shape() -> None:
    mask = torch.zeros(100, 80)
    mask[:50, :40] = 1.0
    result = compute_logits_from_mask(mask)
    assert result.shape == (1, 256, 256)


def test_compute_logits_shape_small_mask() -> None:
    mask = torch.zeros(10, 10)
    result = compute_logits_from_mask(mask)
    assert result.shape == (1, 256, 256)


def test_compute_logits_padding() -> None:
    # Non-square mask: 200x100 -> scale to 256x128, pad width to 256
    mask = torch.zeros(200, 100)
    result = compute_logits_from_mask(mask)
    assert result.shape == (1, 256, 256)


# --- draw_mask tests ---


def test_draw_mask_output_mode() -> None:
    mask = Image.new("L", (50, 50), 255)
    image = Image.new("RGB", (50, 50), (100, 100, 100))
    result = draw_mask(mask, image)
    assert result.mode == "RGBA"


def test_draw_mask_dimensions() -> None:
    mask = Image.new("L", (80, 60), 0)
    image = Image.new("RGB", (80, 60), (0, 0, 0))
    result = draw_mask(mask, image)
    assert result.size == (80, 60)


def test_draw_mask_alpha_varies_with_mask() -> None:
    # Left half = foreground (255), right half = background (0)
    mask = Image.new("L", (100, 100), 0)
    for y in range(100):
        for x in range(50):
            mask.putpixel((x, y), 255)
    image = Image.new("RGB", (100, 100), (100, 100, 100))
    result = draw_mask(mask, image)

    # Foreground pixel (x=10) should have red tint with nonzero alpha overlay
    fg_pixel = result.getpixel((10, 50))
    # Background pixel (x=90) should be untouched
    bg_pixel = result.getpixel((90, 50))
    # Foreground red channel should be higher than background red channel
    assert fg_pixel[0] > bg_pixel[0]
    # Foreground should be semi-transparent (not fully opaque red)
    assert fg_pixel[0] < 255
