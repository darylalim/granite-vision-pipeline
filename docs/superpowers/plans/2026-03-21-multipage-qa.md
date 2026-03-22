# Multipage QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Streamlit page for general-purpose visual question answering across up to 8 document pages using granite-vision-3.3-2b.

**Architecture:** New pipeline module `pipeline/qa.py` with three public functions (`create_qa_model`, `resize_for_qa`, `generate_qa_response`) and a new Streamlit page `pages/qa.py`. Follows the same separation pattern as segmentation and doctags — pipeline handles logic, page handles UI.

**Tech Stack:** transformers (AutoProcessor, AutoModelForVision2Seq), torch, pypdfium2, Pillow, streamlit

**Spec:** `docs/superpowers/specs/2026-03-21-multipage-qa-design.md`

---

### Task 1: `resize_for_qa` — image resizing helper

**Files:**
- Create: `tests/test_qa.py`
- Create: `pipeline/qa.py`

- [ ] **Step 1: Write failing tests for `resize_for_qa`**

```python
"""Tests for the QA module."""

from PIL import Image

from pipeline.qa import resize_for_qa


# --- resize_for_qa tests ---


def test_resize_landscape_image() -> None:
    image = Image.new("RGB", (1024, 768))
    result = resize_for_qa(image)
    assert result.size == (768, 576)


def test_resize_portrait_image() -> None:
    image = Image.new("RGB", (768, 1024))
    result = resize_for_qa(image)
    assert result.size == (576, 768)


def test_small_image_unchanged() -> None:
    image = Image.new("RGB", (400, 300))
    result = resize_for_qa(image)
    assert result.size == (400, 300)


def test_exact_max_dim_unchanged() -> None:
    image = Image.new("RGB", (768, 500))
    result = resize_for_qa(image)
    assert result.size == (768, 500)


def test_resize_preserves_aspect_ratio() -> None:
    image = Image.new("RGB", (1600, 1200))
    result = resize_for_qa(image)
    w, h = result.size
    assert abs(w / h - 1600 / 1200) < 0.01


def test_resize_custom_max_dim() -> None:
    image = Image.new("RGB", (1024, 768))
    result = resize_for_qa(image, max_dim=512)
    assert result.size == (512, 384)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_qa.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.qa'`

- [ ] **Step 3: Implement `resize_for_qa`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_qa.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_qa.py pipeline/qa.py
git commit -m "feat(qa): add resize_for_qa image resizing helper with tests"
```

---

### Task 2: `create_qa_model` — model factory

**Files:**
- Modify: `tests/test_qa.py`
- Modify: `pipeline/qa.py`

- [ ] **Step 1: Write failing tests for `create_qa_model`**

Add the following imports to the **top** of `tests/test_qa.py` (alongside existing imports):

```python
from unittest.mock import MagicMock, patch

from pipeline.qa import create_qa_model
```

Then append the following test functions to the end of the file:

```python
# --- create_qa_model tests ---


@patch("pipeline.qa.AutoModelForVision2Seq")
@patch("pipeline.qa.AutoProcessor")
def test_create_qa_model_loads_correct_model(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    processor, model = create_qa_model(device="cpu")

    mock_processor_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-vision-3.3-2b"
    )
    mock_model_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-vision-3.3-2b"
    )
    assert processor is mock_processor_cls.from_pretrained.return_value
    assert model is mock_model_cls.from_pretrained.return_value.to.return_value


@patch("pipeline.qa.AutoModelForVision2Seq")
@patch("pipeline.qa.AutoProcessor")
def test_create_qa_model_moves_to_device(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    create_qa_model(device="cpu")

    mock_model_cls.from_pretrained.return_value.to.assert_called_once_with("cpu")
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_qa.py::test_create_qa_model_loads_correct_model tests/test_qa.py::test_create_qa_model_moves_to_device -v`
Expected: FAIL with `ImportError` (function doesn't exist yet)

- [ ] **Step 3: Implement `create_qa_model`**

Add imports and function to `pipeline/qa.py`:

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def create_qa_model(
    device: str | None = None,
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load Granite Vision 3.3 2B for multipage QA.

    When device is None, auto-detects: CUDA if available, else CPU.
    MPS is excluded for consistency with other pipeline models.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "ibm-granite/granite-vision-3.3-2b"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
    return processor, model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_qa.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_qa.py pipeline/qa.py
git commit -m "feat(qa): add create_qa_model factory with tests"
```

---

### Task 3: `generate_qa_response` — core inference

**Files:**
- Modify: `tests/test_qa.py`
- Modify: `pipeline/qa.py`

- [ ] **Step 1: Write failing tests for `generate_qa_response`**

Add the following imports to the **top** of `tests/test_qa.py` (alongside existing imports):

```python
import pytest
import torch
from PIL import Image

from pipeline.qa import generate_qa_response
```

Then append the following test functions to the end of the file:

```python
# --- generate_qa_response tests ---


def test_generate_qa_response_rejects_empty_images() -> None:
    with pytest.raises(ValueError, match="1 to 8"):
        generate_qa_response([], "What is this?", MagicMock(), MagicMock())


def test_generate_qa_response_rejects_more_than_8_images() -> None:
    images = [Image.new("RGB", (100, 100)) for _ in range(9)]
    with pytest.raises(ValueError, match="1 to 8"):
        generate_qa_response(images, "What is this?", MagicMock(), MagicMock())


def test_generate_qa_response_prompt_structure() -> None:
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]])
    }

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_processor.decode.return_value = "The answer is 42."

    images = [Image.new("RGB", (100, 100)) for _ in range(3)]
    result = generate_qa_response(images, "What is the answer?", mock_processor, mock_model)

    # Verify conversation structure
    call_args = mock_processor.apply_chat_template.call_args
    conversation = call_args[0][0]
    content = conversation[0]["content"]

    # Should have 3 image entries + 1 text entry
    image_entries = [c for c in content if c["type"] == "image"]
    text_entries = [c for c in content if c["type"] == "text"]
    assert len(image_entries) == 3
    assert len(text_entries) == 1
    assert text_entries[0]["text"] == "What is the answer?"

    # Verify each image entry has an "image" key
    for entry in image_entries:
        assert "image" in entry

    # Verify apply_chat_template keyword arguments
    call_kwargs = mock_processor.apply_chat_template.call_args[1]
    assert call_kwargs["add_generation_prompt"] is True
    assert call_kwargs["tokenize"] is True
    assert call_kwargs["return_dict"] is True
    assert call_kwargs["return_tensors"] == "pt"

    assert result == "The answer is 42."


def test_generate_qa_response_trims_input_and_uses_skip_special_tokens() -> None:
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    input_ids = torch.tensor([[1, 2]])
    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": input_ids
    }

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    mock_processor.decode.return_value = "answer"

    generate_qa_response(
        [Image.new("RGB", (10, 10))], "question", mock_processor, mock_model
    )

    # Verify decode is called with only the new tokens (trimmed)
    mock_processor.decode.assert_called_once()
    decoded_tensor = mock_processor.decode.call_args[0][0]
    assert torch.equal(decoded_tensor, torch.tensor([3, 4]))
    assert mock_processor.decode.call_args[1]["skip_special_tokens"] is True


def test_generate_qa_response_returns_empty_on_no_new_tokens() -> None:
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    input_ids = torch.tensor([[1, 2]])
    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": input_ids
    }

    # Model generates no new tokens (output same length as input)
    mock_model.generate.return_value = torch.tensor([[1, 2]])
    mock_processor.decode.return_value = ""

    result = generate_qa_response(
        [Image.new("RGB", (10, 10))], "question", mock_processor, mock_model
    )
    assert result == ""
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_qa.py::test_generate_qa_response_rejects_empty_images -v`
Expected: FAIL with `ImportError` (function doesn't exist yet)

- [ ] **Step 3: Implement `generate_qa_response`**

Add to `pipeline/qa.py`:

```python
def generate_qa_response(
    images: list[Image.Image],
    question: str,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
) -> str:
    """Answer a question about one or more page images.

    Accepts 1-8 images. Each image is converted to RGB and resized so the
    longer dimension is at most 768px. All images are passed to the model
    in a single conversation turn.

    Raises ValueError if images list has 0 or more than 8 items.
    Returns empty string if the model produces no output.
    """
    if not (1 <= len(images) <= 8):
        raise ValueError(f"Expected 1 to 8 images, got {len(images)}")

    prepared = [resize_for_qa(img.convert("RGB")) for img in images]

    device = next(model.parameters()).device

    content: list[dict] = [{"type": "image", "image": img} for img in prepared]
    content.append({"type": "text", "text": question})

    conversation = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(  # type: ignore[operator]
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=1024)

    trimmed = output[:, inputs["input_ids"].shape[1] :]
    decoded = processor.decode(trimmed[0], skip_special_tokens=True)  # type: ignore[operator]
    return decoded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_qa.py -v`
Expected: All 14 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_qa.py pipeline/qa.py
git commit -m "feat(qa): add generate_qa_response with multi-image prompt and tests"
```

---

### Task 4: Export public API from `pipeline/__init__.py`

**Files:**
- Modify: `pipeline/__init__.py`

- [ ] **Step 1: Add QA imports and exports**

Add to `pipeline/__init__.py`:

```python
from pipeline.qa import create_qa_model, generate_qa_response, resize_for_qa
```

And add to `__all__`:

```python
"create_qa_model",
"generate_qa_response",
"resize_for_qa",
```

- [ ] **Step 2: Verify imports work**

Run: `uv run python -c "from pipeline import create_qa_model, generate_qa_response, resize_for_qa; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run all tests to verify nothing is broken**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add pipeline/__init__.py
git commit -m "feat(qa): export QA public API from pipeline"
```

---

### Task 5: Streamlit UI page

**Files:**
- Create: `pages/qa.py`

- [ ] **Step 1: Create the QA page**

```python
import tempfile
import time
from pathlib import Path

import pypdfium2
import streamlit as st
from PIL import Image

from pipeline import create_qa_model, generate_qa_response, render_pdf_pages

qa_model = st.cache_resource(create_qa_model)

st.title("Multipage QA (Experimental)")
st.write(
    "Ask questions about document pages using IBM Granite Vision. "
    "Upload a PDF or up to 8 images, then type your question."
)

uploaded_files = st.file_uploader(
    "Upload file(s)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

page_images: list[Image.Image] = []
tmp_path: str | None = None
is_pdf = False
selected: list[int] = []

if uploaded_files:
    is_pdf = len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".pdf")

    if is_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_files[0].read())
            tmp_path = tmp_file.name

        pdf = pypdfium2.PdfDocument(tmp_path)
        total_pages = len(pdf)
        pdf.close()

        default_pages = list(range(1, min(9, total_pages + 1)))
        selected = st.multiselect(
            "Select pages (up to 8)",
            options=list(range(1, total_pages + 1)),
            default=default_pages,
            max_selections=8,
        )
    else:
        if len(uploaded_files) > 8:
            st.warning("More than 8 images uploaded. Using the first 8.")
            uploaded_files = uploaded_files[:8]

question = st.text_input("Question", placeholder="e.g., What is shown on these pages?")

has_input = bool(uploaded_files) and bool(question)
if uploaded_files and len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".pdf"):
    has_input = has_input and bool(selected)

if st.button("Answer", type="primary", disabled=not has_input):
    assert uploaded_files is not None
    processor, model = qa_model()

    try:
        if is_pdf:
            assert tmp_path is not None
            with st.spinner("Rendering selected pages..."):
                all_pages = render_pdf_pages(tmp_path)
                page_images = [all_pages[i - 1] for i in selected]
        else:
            page_images = [Image.open(f).convert("RGB") for f in uploaded_files]

        with st.spinner("Generating answer..."):
            start = time.perf_counter_ns()
            answer = generate_qa_response(page_images, question, processor, model)
            duration_s = (time.perf_counter_ns() - start) / 1e9

        if not answer:
            st.warning("Model produced no output.")
        else:
            col_thumbs, col_answer = st.columns([1, 2])
            with col_thumbs:
                for i, img in enumerate(page_images, 1):
                    st.image(img, caption=f"Page {i}", use_container_width=True)
            with col_answer:
                st.markdown(answer)

            st.metric("Duration (s)", f"{duration_s:.2f}")

    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
```

- [ ] **Step 2: Verify page loads without errors**

Run: `uv run python -c "import pages.qa; print('OK')"`
Note: This will fail because of `st.cache_resource` outside a Streamlit context — that's expected. The important check is that the import path resolves. Alternatively, verify with lint:

Run: `uv run ruff check pages/qa.py`
Expected: No errors (or only Streamlit-specific warnings)

- [ ] **Step 3: Run all tests to verify nothing is broken**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add pages/qa.py
git commit -m "feat(qa): add multipage QA Streamlit page"
```

---

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Project Overview**

Add item 4 to the capabilities list:

```markdown
4. **Multipage QA** — answer questions across up to 8 document pages using [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) with images resized to 768px max dimension
```

- [ ] **Step 2: Update Architecture > Pipeline section**

Add entry for `pipeline/qa.py`:

```markdown
- `pipeline/qa.py` — `create_qa_model()` factory, `resize_for_qa()` image resizing, `generate_qa_response()` multi-image QA inference
```

Update `pipeline/__init__.py` entry to include QA exports:

```markdown
- `pipeline/__init__.py` — re-exports public API (`convert`, `create_converter`, `build_output`, `get_description`, `get_table_content`, `create_granite_model`, `create_sam_model`, `segment`, `draw_mask`, `create_doctags_model`, `generate_doctags`, `parse_doctags`, `export_markdown`, `render_pdf_pages`, `create_qa_model`, `resize_for_qa`, `generate_qa_response`)
```

- [ ] **Step 3: Update Architecture > UI section**

Add entry:

```markdown
- `pages/qa.py` — multipage QA page; PDF/image upload, page selection, question input, answer display with thumbnails
```

- [ ] **Step 4: Update Key Details section**

Add:

```markdown
- Multipage QA loads its own Granite Vision instance (independent from segmentation), uses the segmentation-style prompt approach (inline images in conversation dict + `apply_chat_template` with `tokenize=True`)
- QA images are resized so the longer dimension is 768px to stay within GPU memory limits for up to 8 pages
- PDF page count is obtained via `pypdfium2.PdfDocument` without rendering; only selected pages are rendered
```

- [ ] **Step 5: Update Tests section**

Add:

```markdown
- `tests/test_qa.py` — `resize_for_qa()` dimension and aspect ratio tests, `create_qa_model()` with mocked model, `generate_qa_response()` prompt structure, validation, and decode behavior; no model weights required
```

- [ ] **Step 6: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with multipage QA feature"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS (including all new `test_qa.py` tests)

- [ ] **Step 2: Run lint and type checks**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean

- [ ] **Step 3: Review all changes**

Run: `git log --oneline -6`
Expected: 6 commits for tasks 1-6
