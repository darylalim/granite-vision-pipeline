# Image Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add standalone image segmentation using Granite Vision's referring segmentation + SAM refinement, with a dedicated Streamlit page.

**Architecture:** New `pipeline/segmentation.py` module containing all segmentation logic (parsing, mask processing, model inference). New `pages/segmentation.py` for the Streamlit UI. TDD for all testable helpers; model-dependent functions implemented without unit tests.

**Tech Stack:** Python 3.12+, torch, transformers (AutoProcessor, AutoModelForVision2Seq, SamModel, SamProcessor), PIL, Streamlit

**Spec:** `docs/superpowers/specs/2026-03-20-image-segmentation-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `pipeline/segmentation.py` | All segmentation logic: model loading, inference, RLE parsing, mask processing, SAM refinement, visualization |
| `pipeline/__init__.py` | Re-export public segmentation API |
| `tests/test_segmentation.py` | Unit tests for all testable helpers |
| `pages/segmentation.py` | Streamlit UI page for segmentation |
| `pyproject.toml` | Add torch, transformers dependencies |
| `streamlit_app.py` | Add `st.set_page_config` for multipage nav |
| `CLAUDE.md` | Update architecture docs |

---

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml:6-9`

- [ ] **Step 1: Add torch and transformers to project dependencies**

In `pyproject.toml`, update the `dependencies` list:

```toml
dependencies = [
    "docling[vlm]",
    "streamlit",
    "torch",
    "transformers",
]
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: resolves successfully (both are already installed as transitive deps of docling[vlm])

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add torch and transformers as explicit dependencies"
```

---

### Task 2: `extract_segmentation()` — test + implement

**Files:**
- Create: `pipeline/segmentation.py`
- Create: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_segmentation.py`:

```python
"""Tests for the segmentation module."""

from pipeline.segmentation import extract_segmentation


# --- extract_segmentation tests ---


def test_extract_segmentation_valid_text() -> None:
    text = (
        "<seg>others *3\n"
        " dog *1| others *2\n"
        " others *3</seg>"
    )
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_segmentation'`

- [ ] **Step 3: Implement `extract_segmentation()`**

Create `pipeline/segmentation.py`:

```python
"""Image segmentation using Granite Vision and SAM refinement."""

import re


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Lint**

Run: `uv run ruff check pipeline/segmentation.py tests/test_segmentation.py && uv run ruff format --check pipeline/segmentation.py tests/test_segmentation.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/segmentation.py tests/test_segmentation.py
git commit -m "feat: add extract_segmentation with RLE parsing"
```

---

### Task 3: `prepare_mask()` — test + implement

**Files:**
- Modify: `pipeline/segmentation.py`
- Modify: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_segmentation.py`:

```python
import torch

from pipeline.segmentation import prepare_mask


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py::test_prepare_mask_shape -v`
Expected: FAIL — `ImportError: cannot import name 'prepare_mask'`

- [ ] **Step 3: Implement `prepare_mask()`**

Add to `pipeline/segmentation.py`:

```python
import torch
import torch.nn.functional as F


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Lint**

Run: `uv run ruff check pipeline/segmentation.py tests/test_segmentation.py && uv run ruff format --check pipeline/segmentation.py tests/test_segmentation.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/segmentation.py tests/test_segmentation.py
git commit -m "feat: add prepare_mask for upscaling segmentation patches"
```

---

### Task 4: `sample_points()` — test + implement

**Files:**
- Modify: `pipeline/segmentation.py`
- Modify: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_segmentation.py`:

```python
from pipeline.segmentation import sample_points


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py::test_sample_points_counts -v`
Expected: FAIL — `ImportError: cannot import name 'sample_points'`

- [ ] **Step 3: Implement `sample_points()`**

Add to `pipeline/segmentation.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -v`
Expected: all 13 tests PASS

- [ ] **Step 5: Lint**

Run: `uv run ruff check pipeline/segmentation.py tests/test_segmentation.py && uv run ruff format --check pipeline/segmentation.py tests/test_segmentation.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/segmentation.py tests/test_segmentation.py
git commit -m "feat: add sample_points for SAM prompt generation"
```

---

### Task 5: `compute_logits_from_mask()` — test + implement

**Files:**
- Modify: `pipeline/segmentation.py`
- Modify: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_segmentation.py`:

```python
from pipeline.segmentation import compute_logits_from_mask


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py::test_compute_logits_shape -v`
Expected: FAIL — `ImportError: cannot import name 'compute_logits_from_mask'`

- [ ] **Step 3: Implement `compute_logits_from_mask()`**

Add to `pipeline/segmentation.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -v`
Expected: all 16 tests PASS

- [ ] **Step 5: Lint**

Run: `uv run ruff check pipeline/segmentation.py tests/test_segmentation.py && uv run ruff format --check pipeline/segmentation.py tests/test_segmentation.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/segmentation.py tests/test_segmentation.py
git commit -m "feat: add compute_logits_from_mask for SAM input preparation"
```

---

### Task 6: `draw_mask()` — test + implement

**Files:**
- Modify: `pipeline/segmentation.py`
- Modify: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_segmentation.py`:

```python
from PIL import Image

from pipeline.segmentation import draw_mask


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py::test_draw_mask_output_mode -v`
Expected: FAIL — `ImportError: cannot import name 'draw_mask'`

- [ ] **Step 3: Implement `draw_mask()`**

Add to `pipeline/segmentation.py`:

```python
from PIL import Image


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -v`
Expected: all 19 tests PASS

- [ ] **Step 5: Lint**

Run: `uv run ruff check pipeline/segmentation.py tests/test_segmentation.py && uv run ruff format --check pipeline/segmentation.py tests/test_segmentation.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/segmentation.py tests/test_segmentation.py
git commit -m "feat: add draw_mask for segmentation overlay visualization"
```

---

### Task 7: Model loaders, `refine_with_sam()`, and `segment()`

These functions require model weights and cannot be unit tested in the default suite. Implement them directly.

**Files:**
- Modify: `pipeline/segmentation.py`

- [ ] **Step 1: Implement `create_granite_model()`**

Add to `pipeline/segmentation.py`:

```python
from transformers import AutoModelForVision2Seq, AutoProcessor


def create_granite_model(
    device: str | None = None,
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load Granite Vision 3.3 2B for segmentation.

    When device is None, auto-detects: CUDA if available, else CPU.
    MPS is excluded due to limited operator support in SAM/transformers.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "ibm-granite/granite-vision-3.3-2b"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
    return processor, model
```

- [ ] **Step 2: Implement `create_sam_model()`**

Add to `pipeline/segmentation.py`:

```python
from transformers import SamModel, SamProcessor


def create_sam_model(
    device: str | None = None,
) -> tuple[SamProcessor, SamModel]:
    """Load SAM ViT-Huge for mask refinement.

    When device is None, auto-detects: CUDA if available, else CPU.
    MPS is excluded due to limited operator support in SAM/transformers.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "facebook/sam-vit-huge"
    processor = SamProcessor.from_pretrained(model_path)
    model = SamModel.from_pretrained(model_path).to(device)
    return processor, model
```

- [ ] **Step 3: Implement `refine_with_sam()`**

Add to `pipeline/segmentation.py`:

```python
def refine_with_sam(
    mask: torch.Tensor,
    image: Image.Image,
    sam: tuple[SamProcessor, SamModel],
) -> torch.Tensor:
    """Run SAM inference to refine a coarse mask.

    Returns refined binary mask tensor at original image resolution.
    """
    sam_processor, sam_model = sam
    device = next(sam_model.parameters()).device

    input_points, input_labels = sample_points(mask)
    logits = compute_logits_from_mask(mask)

    sam_inputs = sam_processor(
        image,
        input_points=input_points.unsqueeze(0).float().numpy(),
        input_labels=input_labels.unsqueeze(0).numpy(),
        return_tensors="pt",
    ).to(device)

    image_positional_embeddings = sam_model.get_image_wide_positional_embeddings()

    with torch.inference_mode():
        embeddings = sam_model.get_image_embeddings(sam_inputs["pixel_values"])
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            input_points=sam_inputs["input_points"],
            input_labels=sam_inputs["input_labels"],
            input_masks=logits.unsqueeze(0).to(device),
            input_boxes=None,
        )
        segmentation_maps, _, _ = sam_model.mask_decoder(
            image_embeddings=embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

    post_processed = sam_processor.post_process_masks(
        segmentation_maps.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )
    # post_process_masks returns logits; threshold at 0.0 for binary mask
    return (post_processed[0].squeeze() > 0.0).to(torch.uint8)
```

- [ ] **Step 4: Implement `segment()`**

Add to `pipeline/segmentation.py`:

```python
def segment(
    image: Image.Image,
    prompt: str,
    granite: tuple[AutoProcessor, AutoModelForVision2Seq],
    sam: tuple[SamProcessor, SamModel],
) -> Image.Image | None:
    """Run full segmentation pipeline.

    Converts input to RGB. Returns mask as PIL Image (mode "L",
    0=background, 255=foreground) or None if no <seg> tags found.
    """
    image = image.convert("RGB")
    granite_processor, granite_model = granite
    device = next(granite_model.parameters()).device

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"seg: Could you segment the '{prompt}' in the image? "
                    "Respond with the segmentation mask",
                },
            ],
        },
    ]

    inputs = granite_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = granite_model.generate(**inputs, max_new_tokens=8192)

    decoded = granite_processor.decode(output[0], skip_special_tokens=True)

    flat_mask = extract_segmentation(decoded)
    if flat_mask is None:
        return None

    coarse_mask = prepare_mask(flat_mask, patch_h=24, patch_w=24, size=image.size)
    refined_mask = refine_with_sam(coarse_mask, image, sam)

    pil_mask = Image.fromarray((refined_mask * 255).numpy(), mode="L")
    return pil_mask
```

- [ ] **Step 5: Lint**

Run: `uv run ruff check pipeline/segmentation.py && uv run ruff format --check pipeline/segmentation.py`
Expected: no errors

- [ ] **Step 6: Run all existing tests to verify nothing broke**

Run: `uv run pytest -v`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add pipeline/segmentation.py
git commit -m "feat: add segment, refine_with_sam, and model loaders"
```

---

### Task 8: Update `pipeline/__init__.py` exports

**Files:**
- Modify: `pipeline/__init__.py:1-10`

- [ ] **Step 1: Add segmentation exports**

Update `pipeline/__init__.py`:

```python
from pipeline.config import convert, create_converter
from pipeline.output import build_output, get_description, get_table_content
from pipeline.segmentation import create_granite_model, create_sam_model, draw_mask, segment

__all__ = [
    "build_output",
    "convert",
    "create_converter",
    "create_granite_model",
    "create_sam_model",
    "draw_mask",
    "get_description",
    "get_table_content",
    "segment",
]
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest -v`
Expected: all tests PASS

- [ ] **Step 3: Lint**

Run: `uv run ruff check pipeline/__init__.py && uv run ruff format --check pipeline/__init__.py`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add pipeline/__init__.py
git commit -m "feat: export segmentation public API from pipeline"
```

---

### Task 9: Streamlit UI page + page config

**Files:**
- Create: `pages/segmentation.py`
- Modify: `streamlit_app.py:1-14`

- [ ] **Step 1: Create `pages/` directory**

Run: `mkdir -p pages`

- [ ] **Step 2: Add `st.set_page_config` to `streamlit_app.py`**

`st.set_page_config` must be the first Streamlit command that writes to the page. Insert it after imports/cache setup but before `st.title`.

Replace in `streamlit_app.py`:

```python
converter = st.cache_resource(create_converter)

st.title("Granite Vision Pipeline")
```

With:

```python
converter = st.cache_resource(create_converter)

st.set_page_config(page_title="Granite Vision Pipeline")
st.title("Granite Vision Pipeline")
```

- [ ] **Step 3: Create `pages/segmentation.py`**

```python
import io

import streamlit as st
from PIL import Image

from pipeline import create_granite_model, create_sam_model, draw_mask, segment

granite_model = st.cache_resource(create_granite_model)
sam_model = st.cache_resource(create_sam_model)

st.title("Image Segmentation (Experimental)")
st.write(
    "Segment objects in images using natural language prompts. "
    "Powered by Granite Vision with SAM refinement."
)

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
prompt = st.text_input("Segmentation prompt", placeholder="e.g., the dog on the left")

if st.button("Segment", type="primary", disabled=not uploaded_file or not prompt):
    assert uploaded_file is not None
    image = Image.open(uploaded_file)

    with st.spinner("Running segmentation... This may take a few minutes."):
        mask = segment(
            image,
            prompt,
            granite=granite_model(),
            sam=sam_model(),
        )

    if mask is None:
        st.error("Segmentation failed — no mask found in model output.")
    else:
        col_orig, col_overlay = st.columns(2)
        col_orig.image(image, caption="Original")
        overlay = draw_mask(mask, image)
        col_overlay.image(overlay, caption="Segmentation overlay")

        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        st.download_button(
            label="Download mask",
            data=buf.getvalue(),
            file_name="segmentation_mask.png",
            mime="image/png",
        )
```

- [ ] **Step 4: Lint**

Run: `uv run ruff check pages/segmentation.py streamlit_app.py && uv run ruff format --check pages/segmentation.py streamlit_app.py`
Expected: no errors

- [ ] **Step 5: Run all tests to ensure nothing broke**

Run: `uv run pytest -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add pages/segmentation.py streamlit_app.py
git commit -m "feat: add segmentation Streamlit page with multipage navigation"
```

---

### Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md architecture and test sections**

Add to the Architecture section, after the `streamlit_app.py` bullet:

```markdown
- `pipeline/segmentation.py` — `segment()` runs Granite Vision referring segmentation + SAM refinement; `draw_mask()` for visualization; `create_granite_model()` and `create_sam_model()` factories; internal helpers for RLE parsing, mask processing, point sampling, and logit computation
- `pages/segmentation.py` — standalone segmentation UI page; image upload, text prompt, mask overlay preview, mask download; models cached via `st.cache_resource`
```

Add to the Dependencies section under Runtime:

```markdown
- `torch` — tensor operations and model inference for segmentation
- `transformers` — model loading (Granite Vision, SAM) for segmentation
```

Add to the Key details section:

```markdown
- Segmentation loads separate Granite Vision and SAM model instances (not shared with docling's internal model)
- Adding `pages/` directory activates Streamlit multipage navigation with sidebar
```

Add to the Tests section:

```markdown
- `tests/test_segmentation.py` — `extract_segmentation()`, `prepare_mask()`, `sample_points()`, `compute_logits_from_mask()`, and `draw_mask()` unit tests; no model weights required
```

- [ ] **Step 2: Lint and verify**

Run: `uv run pytest -v`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with segmentation architecture"
```
