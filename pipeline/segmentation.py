"""Image segmentation using Granite Vision and SAM refinement."""

import re

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, SamModel, SamProcessor


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
    try:
        rows = match.group(1).strip().split("\n")
        tokens = [token.split(" *") for row in rows for token in row.split("| ")]
        tokens = [x[0].strip() for x in tokens for _ in range(int(x[1]))]
    except (IndexError, ValueError):
        return None

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
    t = (
        F.interpolate(
            t[None, None],
            size=(size[1], size[0]),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
    )
    return t


def _sample_points_from_mask(
    mask: torch.Tensor,
    num_points: int,
    is_positive: bool,
    generator: torch.Generator | None = None,
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
        low=0,
        high=len(target_indices),
        size=(num_points,),
        device=mask.device,
        generator=generator,
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
    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=mask.device)
        generator.manual_seed(seed)

    pos_coords = _sample_points_from_mask(mask, num_pos, True, generator)
    neg_coords = _sample_points_from_mask(mask, num_neg, False, generator)

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
