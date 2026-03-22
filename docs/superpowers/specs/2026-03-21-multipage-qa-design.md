# Multipage QA — Design Spec

## Overview

Add a new Streamlit page for general-purpose visual question answering (VQA) across multiple consecutive document pages using `ibm-granite/granite-vision-3.3-2b`. The model was trained to handle QA tasks using up to 8 consecutive pages. To stay within GPU memory limits, images are resized so their longer dimension is 768 pixels. Users upload a PDF (selecting pages) or multiple images, type a question, and receive a text answer alongside thumbnails of the pages the model was given.

## Architecture

Follows the established pattern: core logic in `pipeline/qa.py`, UI in `pages/qa.py`.

### New Files

- `pipeline/qa.py` — model loading, image resizing, multi-image QA inference
- `pages/qa.py` — Streamlit UI page
- `tests/test_qa.py` — unit tests for pipeline logic

### Modified Files

- `pipeline/__init__.py` — re-export public API (`create_qa_model`, `resize_for_qa`, `generate_qa_response`)

## Pipeline Module (`pipeline/qa.py`)

### Model Management

- `create_qa_model(device: str | None = None) -> tuple[AutoProcessor, AutoModelForVision2Seq]`
  - Loads `ibm-granite/granite-vision-3.3-2b` via Transformers (`AutoModelForVision2Seq`)
  - Auto-detects device: CUDA if available, else CPU (skip MPS, consistent with segmentation)
  - Returns `(processor, model)` tuple — processor first, matching existing convention
  - Independent instance from segmentation's `create_granite_model()` (not shared)
  - Cached via `st.cache_resource` at the page level

### Image Resizing

- `resize_for_qa(image: Image.Image, max_dim: int = 768) -> Image.Image`
  - If the image's longer dimension exceeds `max_dim`, resize so the longer dimension equals `max_dim`, preserving aspect ratio (use `Image.LANCZOS` resampling)
  - If the image is already within bounds, return it unchanged
  - Ensures the model can handle up to 8 pages without exceeding GPU memory

### Core Inference

- `generate_qa_response(images: list[Image.Image], question: str, processor: AutoProcessor, model: AutoModelForVision2Seq) -> str`
  - Validates `images` has 1–8 items; raises `ValueError` if 0 or more than 8
  - Calls `resize_for_qa` on each image before building the prompt
  - Infers device from model parameters (no separate `device` arg, matching `segment()` pattern)
  - Constructs a single conversation turn with one `{"type": "image"}` entry per page image, followed by `{"type": "text", "text": question}`
  - Runs `model.generate()` with `max_new_tokens=1024` under `torch.inference_mode()`
  - Decodes output, strips input tokens, returns response text
  - Returns empty string on empty model output

## UI Page (`pages/qa.py`)

### Layout

- Title: "Multipage QA (Experimental)"
- Description text explaining the feature (up to 8 pages, general-purpose VQA)
- File uploader accepting `["pdf", "png", "jpg", "jpeg", "webp"]`, multiple files allowed
- For PDF uploads: multiselect widget to choose up to 8 pages (populated after upload, using `render_pdf_pages()` to determine page count)
- For image uploads: accept up to 8 images, show `st.warning` if more than 8 are uploaded (use first 8)
- One PDF or multiple images per run — not both simultaneously
- Text input for the user's question
- "Answer" button (disabled until files + question are provided)

### Results Display

- `st.spinner("Generating answer...")` during inference
- Two-column layout:
  - **Left column**: page thumbnails labeled "Page 1", "Page 2", etc.
  - **Right column**: model's answer rendered as markdown
- Duration metric shown below the answer

### Data Flow

```
PDF upload -> render_pdf_pages() -> select up to 8 pages -> [Image list]
Image upload -> [Image list] (up to 8)

[Image list] + question -> generate_qa_response() -> answer text
```

## Testing (`tests/test_qa.py`)

Unit tests that do not require model weights:

- `resize_for_qa()` — landscape image (1024x768) resized to 768x576, portrait image (768x1024) resized to 576x768, small image (400x300) returned unchanged, aspect ratio preserved
- `generate_qa_response()` — raises `ValueError` for 0 images, raises `ValueError` for 9 images, correct prompt structure verified via mocked processor/model (each image appears as `{"type": "image"}` entry, question appears as `{"type": "text"}`), empty model output returns empty string
- `create_qa_model()` — mocked to verify correct model ID (`ibm-granite/granite-vision-3.3-2b`) loaded, device placement verified

## Dependencies

No new dependencies. All required packages are already in the project:

- `transformers` — model loading (`AutoProcessor`, `AutoModelForVision2Seq`)
- `torch` — inference (`torch.inference_mode()`)
- `pypdfium2` — PDF page rendering (reuses `render_pdf_pages` from `pipeline.doctags`)
- `Pillow` — image resizing (transitive via existing deps)

## Model Details

- Model: `ibm-granite/granite-vision-3.3-2b`
- Same model used in segmentation, but loaded as an independent instance
- Multipage training: up to 8 consecutive pages per context
- Image constraint: longer dimension resized to 768px to manage GPU memory
- `max_new_tokens`: 1024 (sufficient for QA answers)
