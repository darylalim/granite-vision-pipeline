# Refactoring Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce code duplication and improve memory efficiency by extracting shared helpers for model loading, text generation, temp files, and timing.

**Architecture:** New `pipeline/models.py` consolidates vision model factories and the generate-trim-decode pattern. New `pipeline/utils.py` provides `temp_upload` and `timed` context managers. Existing modules become thin wrappers. A `get_pdf_page_count` helper eliminates a redundant PDF open in the QA page.

**Tech Stack:** Python, PyTorch, Transformers, Streamlit, pypdfium2, sentence-transformers, ChromaDB, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-refactoring-improvements-design.md`

---

### Task 1: Create `pipeline/utils.py` with `temp_upload` and `timed`

**Files:**
- Create: `pipeline/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write tests for `timed` context manager**

```python
"""Tests for the utils module."""

import time

from pipeline.utils import timed


def test_timed_returns_duration() -> None:
    with timed() as t:
        time.sleep(0.01)
    assert t.duration_s >= 0.01


def test_timed_duration_is_zero_before_exit() -> None:
    with timed() as t:
        assert t.duration_s == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.utils'`

- [ ] **Step 3: Write tests for `temp_upload` context manager**

Add to `tests/test_utils.py`:

```python
import io
from pathlib import Path

from pipeline.utils import temp_upload


def test_temp_upload_creates_and_cleans_up_file() -> None:
    uploaded = io.BytesIO(b"test content")
    with temp_upload(uploaded, suffix=".pdf") as path:
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"test content"
        assert path.endswith(".pdf")
    assert not Path(path).exists()


def test_temp_upload_cleans_up_on_exception() -> None:
    uploaded = io.BytesIO(b"data")
    path_ref: str | None = None
    try:
        with temp_upload(uploaded) as path:
            path_ref = path
            raise RuntimeError("simulated error")
    except RuntimeError:
        pass
    assert path_ref is not None
    assert not Path(path_ref).exists()
```

- [ ] **Step 4: Implement `pipeline/utils.py`**

```python
"""Shared utility helpers for the pipeline."""

import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, BinaryIO


@contextmanager
def temp_upload(
    uploaded_file: IO[bytes] | BinaryIO, suffix: str = ".pdf"
) -> Generator[str, None, None]:
    """Write an uploaded file to a temporary path and clean up on exit."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        path = f.name
    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)


class Timer:
    """Mutable container for elapsed time, used by the `timed` context manager."""

    __slots__ = ("duration_s",)

    def __init__(self) -> None:
        self.duration_s: float = 0.0


@contextmanager
def timed() -> Generator[Timer, None, None]:
    """Measure wall-clock time in seconds.

    Usage::

        with timed() as t:
            do_work()
        print(t.duration_s)
    """
    timer = Timer()
    start = time.perf_counter_ns()
    try:
        yield timer
    finally:
        timer.duration_s = (time.perf_counter_ns() - start) / 1e9
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline/utils.py tests/test_utils.py
git commit -m "feat: add temp_upload and timed context managers to pipeline/utils"
```

---

### Task 2: Create `pipeline/models.py` with `_load_vision_model` and model factories

**Files:**
- Create: `pipeline/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write tests for `_load_vision_model`**

```python
"""Tests for the models module."""

from unittest.mock import MagicMock, patch


@patch("pipeline.models.AutoModelForVision2Seq")
@patch("pipeline.models.AutoProcessor")
def test_load_vision_model_loads_and_moves_to_device(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    from pipeline.models import _load_vision_model

    processor, model = _load_vision_model("some/repo", device="cpu")

    mock_processor_cls.from_pretrained.assert_called_once_with("some/repo")
    mock_model_cls.from_pretrained.assert_called_once_with("some/repo")
    mock_model_cls.from_pretrained.return_value.to.assert_called_once_with("cpu")
    assert processor is mock_processor_cls.from_pretrained.return_value
    assert model is mock_model_cls.from_pretrained.return_value.to.return_value
```

- [ ] **Step 2: Write tests for `create_granite_vision_model`**

Add to `tests/test_models.py`:

```python
@patch("pipeline.models.AutoModelForVision2Seq")
@patch("pipeline.models.AutoProcessor")
def test_create_granite_vision_model_uses_correct_repo(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    from pipeline.models import create_granite_vision_model

    create_granite_vision_model(device="cpu")

    mock_processor_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-vision-3.3-2b"
    )
    mock_model_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-vision-3.3-2b"
    )
```

- [ ] **Step 3: Write tests for `create_doctags_model`**

Add to `tests/test_models.py`:

```python
@patch("pipeline.models.AutoModelForVision2Seq")
@patch("pipeline.models.AutoProcessor")
def test_create_doctags_model_uses_correct_repo(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    from pipeline.models import create_doctags_model

    create_doctags_model(device="cpu")

    mock_processor_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-docling-258M"
    )
    mock_model_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-docling-258M"
    )
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.models'`

- [ ] **Step 5: Implement model factories in `pipeline/models.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: All 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add pipeline/models.py tests/test_models.py
git commit -m "feat: add shared model factories in pipeline/models"
```

---

### Task 3: Add `generate_response` to `pipeline/models.py`

**Files:**
- Modify: `pipeline/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write test for `generate_response` basic flow**

Add to `tests/test_models.py`:

```python
import torch


def test_generate_response_trims_and_decodes() -> None:
    from pipeline.models import generate_response

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
    mock_processor.decode.return_value = "Generated text."

    conversation = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    result = generate_response(conversation, mock_processor, mock_model, max_new_tokens=512)

    # Verify apply_chat_template kwargs
    call_kwargs = mock_processor.apply_chat_template.call_args[1]
    assert call_kwargs["add_generation_prompt"] is True
    assert call_kwargs["tokenize"] is True
    assert call_kwargs["return_dict"] is True
    assert call_kwargs["return_tensors"] == "pt"

    # Verify generate called with max_new_tokens
    _, gen_kwargs = mock_model.generate.call_args
    assert gen_kwargs["max_new_tokens"] == 512

    # Verify trimming: input had 3 tokens, output has 5, so decoded tokens are [4, 5]
    decoded_tensor = mock_processor.decode.call_args[0][0]
    assert torch.equal(decoded_tensor, torch.tensor([4, 5]))
    assert mock_processor.decode.call_args[1]["skip_special_tokens"] is True

    assert result == "Generated text."


def test_generate_response_empty_output() -> None:
    from pipeline.models import generate_response

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

    # Model returns same length as input (no new tokens)
    mock_model.generate.return_value = torch.tensor([[1, 2]])
    mock_processor.decode.return_value = ""

    conversation = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    result = generate_response(conversation, mock_processor, mock_model)
    assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py::test_generate_response_trims_and_decodes tests/test_models.py::test_generate_response_empty_output -v`
Expected: FAIL — `ImportError: cannot import name 'generate_response'`

- [ ] **Step 3: Implement `generate_response`**

Add to `pipeline/models.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/models.py tests/test_models.py
git commit -m "feat: add generate_response helper to pipeline/models"
```

---

### Task 4: Add `get_pdf_page_count` to `pipeline/doctags.py`

**Files:**
- Modify: `pipeline/doctags.py`
- Modify: `tests/test_doctags.py`

- [ ] **Step 1: Write test for `get_pdf_page_count`**

Add to `tests/test_doctags.py` after the existing imports:

```python
from pipeline.doctags import get_pdf_page_count
```

Add test after the `render_pdf_pages` tests section:

```python
# --- get_pdf_page_count tests ---


def test_get_pdf_page_count_returns_correct_count() -> None:
    all_pages = render_pdf_pages(TEST_PDF)
    count = get_pdf_page_count(TEST_PDF)
    assert count == len(all_pages)


def test_get_pdf_page_count_positive() -> None:
    assert get_pdf_page_count(TEST_PDF) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_doctags.py::test_get_pdf_page_count_returns_correct_count tests/test_doctags.py::test_get_pdf_page_count_positive -v`
Expected: FAIL — `ImportError: cannot import name 'get_pdf_page_count'`

- [ ] **Step 3: Implement `get_pdf_page_count`**

Add to `pipeline/doctags.py` after the `render_pdf_pages` function:

```python
def get_pdf_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF without rendering."""
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        return len(pdf)
    finally:
        pdf.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_doctags.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/doctags.py tests/test_doctags.py
git commit -m "feat: add get_pdf_page_count helper to pipeline/doctags"
```

---

### Task 5: Migrate `pipeline/qa.py` to use shared helpers

**Files:**
- Modify: `pipeline/qa.py`
- Modify: `tests/test_qa.py`

- [ ] **Step 1: Update `pipeline/qa.py`**

Replace the entire file with:

```python
"""Multipage QA using Granite Vision."""

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from pipeline.models import generate_response


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
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


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

    content: list[dict] = [{"type": "image", "image": img} for img in prepared]
    content.append({"type": "text", "text": question})

    conversation = [{"role": "user", "content": content}]

    return generate_response(conversation, processor, model, max_new_tokens=1024)
```

- [ ] **Step 2: Update test mock paths in `tests/test_qa.py`**

The `create_qa_model` tests must be removed (the function no longer exists in `qa.py`). The `generate_qa_response` tests need mock paths updated since `generate_response` is now called internally.

In `tests/test_qa.py`, make these changes:

Remove the import of `create_qa_model` from the top-level import:

```python
from pipeline.qa import generate_qa_response, resize_for_qa
```

Remove both `test_create_qa_model_*` tests (lines 61-88).

Update `test_generate_qa_response_prompt_structure` — the function now delegates to `generate_response`, so we mock that instead of inline model calls:

```python
@patch("pipeline.qa.generate_response")
def test_generate_qa_response_prompt_structure(
    mock_gen: MagicMock,
) -> None:
    mock_gen.return_value = "The answer is 42."

    images = [Image.new("RGB", (100, 100)) for _ in range(3)]
    result = generate_qa_response(
        images, "What is the answer?", MagicMock(), MagicMock()
    )

    conversation = mock_gen.call_args[0][0]
    content = conversation[0]["content"]

    image_entries = [c for c in content if c["type"] == "image"]
    text_entries = [c for c in content if c["type"] == "text"]
    assert len(image_entries) == 3
    assert len(text_entries) == 1
    assert text_entries[0]["text"] == "What is the answer?"

    assert mock_gen.call_args[1]["max_new_tokens"] == 1024
    assert result == "The answer is 42."
```

Update `test_generate_qa_response_trims_input_and_uses_skip_special_tokens` — this behavior is now in `generate_response` (already tested in `test_models.py`), so remove this test.

Update `test_generate_qa_response_returns_empty_on_no_new_tokens`:

```python
@patch("pipeline.qa.generate_response")
def test_generate_qa_response_returns_empty_on_no_new_tokens(
    mock_gen: MagicMock,
) -> None:
    mock_gen.return_value = ""
    result = generate_qa_response(
        [Image.new("RGB", (10, 10))], "question", MagicMock(), MagicMock()
    )
    assert result == ""
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_qa.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add pipeline/qa.py tests/test_qa.py
git commit -m "refactor: migrate qa.py to use generate_response from models"
```

---

### Task 6: Migrate `pipeline/search.py` to use `generate_response` and optimize `_chunk_text`

**Files:**
- Modify: `pipeline/search.py`
- Modify: `tests/test_search.py`

- [ ] **Step 1: Update `generate_answer` in `pipeline/search.py`**

Replace the `generate_answer` function (lines 229-278) with:

```python
def generate_answer(
    question: str,
    context: list[dict],
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
) -> str:
    """Generate a RAG answer using retrieved context and Granite Vision.

    Constructs a text-only prompt from the context and question, sends to
    the model via generate_response. Uses max_new_tokens=1024.
    """
    context_lines: list[str] = []
    for i, item in enumerate(context, 1):
        type_label = item["metadata"].get("type", "element")
        source = item["metadata"].get("source", "")
        label = f"[Element {i} - {type_label}"
        if source:
            label += f" from {source}"
        label += "]"
        context_lines.append(f"{label}: {item['text']}")

    context_str = (
        "\n".join(context_lines) if context_lines else "(No context available)"
    )

    prompt = (
        "Use the following context from a document to answer the question.\n"
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}"
    )

    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    return generate_response(conversation, processor, model, max_new_tokens=1024)
```

Update imports at the top of `pipeline/search.py` — add:

```python
from pipeline.models import generate_response
```

Remove the `torch` import (no longer needed in this file) and the `AutoModelForVision2Seq`, `AutoProcessor` imports from `transformers` (still needed for type annotations in `generate_answer` signature). Actually, keep the `transformers` imports for the type annotations. Remove only `torch`.

- [ ] **Step 2: Optimize `_chunk_text`**

Replace the `_chunk_text` function (lines 72-125) with:

```python
def _chunk_text(
    text: str,
    model: SentenceTransformer,
    token_limit: int = 8000,
    chunk_size: int = 7000,
    overlap: int = 200,
) -> list[str]:
    """Split text into chunks if it exceeds the token limit.

    Uses a character-based estimate (len/4) for the initial check, then
    exact tokenization only when building chunks. Splits on sentence
    boundaries (preserving trailing periods) via regex lookbehind, falling
    back to '\\n' for content like table markdown.
    Returns [text] if within token_limit.
    """
    # Quick estimate: ~4 chars per token on average
    if len(text) // 4 <= token_limit:
        return [text]

    # Exact check for borderline cases
    tokenizer = model.tokenizer
    token_count = len(tokenizer.encode(text))
    if token_count <= token_limit:
        return [text]

    # Split on sentence boundaries (preserving the period) or newlines
    parts = re.split(r"(?<=\.) ", text)
    joiner = " "
    if len(parts) <= 1:
        parts = text.split("\n")
        joiner = "\n"
    if len(parts) <= 1:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = len(tokenizer.encode(part))

        if current_tokens + part_tokens > chunk_size and current:
            chunks.append(joiner.join(current))
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for p in reversed(current):
                p_tokens = len(tokenizer.encode(p))
                if overlap_tokens + p_tokens > overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_tokens += p_tokens
            current = overlap_parts
            current_tokens = overlap_tokens

        current.append(part)
        current_tokens += part_tokens

    if current:
        chunks.append(joiner.join(current))

    return chunks if chunks else [text]
```

- [ ] **Step 3: Update `generate_answer` tests in `tests/test_search.py`**

The `generate_answer` tests need to mock `generate_response` instead of inline model calls. Update the tests:

Replace `test_generate_answer_prompt_structure` (keep the prompt-building assertions, mock `generate_response`):

```python
@patch("pipeline.search.generate_response")
def test_generate_answer_prompt_structure(mock_gen: MagicMock) -> None:
    from pipeline.search import generate_answer

    mock_gen.return_value = "The answer."

    context = [
        {
            "text": "Revenue grew 20%.",
            "metadata": {
                "type": "picture",
                "source": "test.pdf",
                "element_number": 1,
                "reference": "#/pictures/0",
            },
            "similarity": 0.8,
        },
        {
            "text": "| Q | Rev |\n|---|---|\n| Q4 | 1.2M |",
            "metadata": {
                "type": "table",
                "source": "test.pdf",
                "element_number": 2,
                "reference": "#/tables/0",
            },
            "similarity": 0.7,
        },
    ]

    mock_processor = MagicMock()
    mock_model = MagicMock()
    result = generate_answer(
        "How did revenue change?", context, mock_processor, mock_model
    )

    conversation = mock_gen.call_args[0][0]
    content = conversation[0]["content"]

    assert len(content) == 1
    assert content[0]["type"] == "text"

    prompt_text = content[0]["text"]
    assert "[Element 1 - picture from test.pdf]" in prompt_text
    assert "[Element 2 - table from test.pdf]" in prompt_text
    assert "Revenue grew 20%." in prompt_text
    assert "How did revenue change?" in prompt_text

    assert mock_gen.call_args[1]["max_new_tokens"] == 1024
    assert result == "The answer."
```

Replace `test_generate_answer_uses_max_new_tokens`:

```python
@patch("pipeline.search.generate_response")
def test_generate_answer_uses_max_new_tokens(mock_gen: MagicMock) -> None:
    from pipeline.search import generate_answer

    mock_gen.return_value = "answer"

    generate_answer(
        "q",
        [{"text": "t", "metadata": {"type": "picture"}, "similarity": 0.8}],
        MagicMock(),
        MagicMock(),
    )

    assert mock_gen.call_args[1]["max_new_tokens"] == 1024
```

Replace `test_generate_answer_empty_context`:

```python
@patch("pipeline.search.generate_response")
def test_generate_answer_empty_context(mock_gen: MagicMock) -> None:
    from pipeline.search import generate_answer

    mock_gen.return_value = "No context available."

    result = generate_answer("question", [], MagicMock(), MagicMock())

    mock_gen.assert_called_once()
    assert result == "No context available."
```

Replace `test_generate_answer_missing_type_defaults_to_element`:

```python
@patch("pipeline.search.generate_response")
def test_generate_answer_missing_type_defaults_to_element(mock_gen: MagicMock) -> None:
    from pipeline.search import generate_answer

    mock_gen.return_value = "answer"

    context = [{"text": "some text", "metadata": {}, "similarity": 0.5}]
    generate_answer("q", context, MagicMock(), MagicMock())

    prompt_text = mock_gen.call_args[0][0][0]["content"][0]["text"]
    assert "[Element 1 - element]" in prompt_text
```

Add the `patch` import at the top of `tests/test_search.py` if not already there (it already imports `from unittest.mock import MagicMock, patch`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/search.py tests/test_search.py
git commit -m "refactor: migrate search.py to use generate_response and optimize _chunk_text"
```

---

### Task 7: Migrate `pipeline/segmentation.py` to use shared helpers

**Files:**
- Modify: `pipeline/segmentation.py`

- [ ] **Step 1: Update `pipeline/segmentation.py`**

Remove the `create_granite_model` function (lines 176-189).

Replace the `segment` function (lines 257-307) — the only change is replacing the inline generate logic with `generate_response`:

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

    decoded = generate_response(
        conversation, granite_processor, granite_model, max_new_tokens=8192
    )

    flat_mask = extract_segmentation(decoded)
    if flat_mask is None:
        return None

    coarse_mask = prepare_mask(flat_mask, patch_h=24, patch_w=24, size=image.size)
    refined_mask = refine_with_sam(coarse_mask, image, sam)

    pil_mask = Image.fromarray((refined_mask * 255).numpy(), mode="L")
    return pil_mask
```

Add the import at the top of `pipeline/segmentation.py`:

```python
from pipeline.models import generate_response
```

Remove `torch` from imports (still needed for other functions in the file — check: `torch` is used by `prepare_mask`, `sample_points`, `compute_logits_from_mask`, `refine_with_sam`, etc.). Keep `torch` import. Remove unused variable `device` from the `segment` function body (was `device = next(granite_model.parameters()).device` — no longer needed since `generate_response` handles this).

- [ ] **Step 2: Run tests to verify nothing is broken**

Run: `uv run pytest tests/test_segmentation.py -v`
Expected: All tests PASS (these test internal helpers, not model loading or the full `segment` function)

- [ ] **Step 3: Commit**

```bash
git add pipeline/segmentation.py
git commit -m "refactor: migrate segmentation.py to use generate_response, remove create_granite_model"
```

---

### Task 8: Remove `create_doctags_model` from `pipeline/doctags.py`

**Files:**
- Modify: `pipeline/doctags.py`
- Modify: `tests/test_doctags.py`

- [ ] **Step 1: Remove `create_doctags_model` from `pipeline/doctags.py`**

Delete the `create_doctags_model` function (lines 88-101) and remove the now-unused `torch` import. Also remove the `AutoModelForVision2Seq` and `AutoProcessor` imports from `transformers` if they are still needed for `generate_doctags` type annotations — check: yes, `generate_doctags` uses `AutoProcessor` and `AutoModelForVision2Seq` as parameter types, so keep those imports. Actually, `torch` is also still used in `generate_doctags` (line 80: `with torch.inference_mode()`), so keep `torch` too.

Just delete the `create_doctags_model` function (lines 88-101).

- [ ] **Step 2: Update `tests/test_doctags.py` mock paths**

The `create_doctags_model` tests need to import from `pipeline.models` and mock the correct path:

```python
from pipeline.models import create_doctags_model
```

Update the mock decorators:

```python
@patch("pipeline.models.AutoModelForVision2Seq")
@patch("pipeline.models.AutoProcessor")
def test_create_doctags_model_loads_correct_model(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    processor, model = create_doctags_model(device="cpu")
    ...


@patch("pipeline.models.AutoModelForVision2Seq")
@patch("pipeline.models.AutoProcessor")
def test_create_doctags_model_moves_to_device(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    create_doctags_model(device="cpu")
    ...
```

Actually, these tests are now redundant with `tests/test_models.py` which already tests `create_doctags_model`. Remove both `test_create_doctags_model_*` tests from `tests/test_doctags.py`. Also remove the `from pipeline.doctags import create_doctags_model` from the imports (it was `from pipeline.doctags import ...` — just remove `create_doctags_model` from that import list).

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_doctags.py tests/test_models.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add pipeline/doctags.py tests/test_doctags.py
git commit -m "refactor: remove create_doctags_model from doctags.py, now in models.py"
```

---

### Task 9: Migrate `streamlit_app.py` to use `temp_upload` and `timed`

**Files:**
- Modify: `streamlit_app.py`

- [ ] **Step 1: Update imports**

Replace:
```python
import tempfile
import time
from pathlib import Path
```

With:
```python
import time
```

Add to the pipeline import block:
```python
from pipeline import (
    build_output,
    convert,
    create_converter,
    create_embedding_model,
    get_collection,
    get_description,
    index_elements,
    temp_upload,
    timed,
)
```

Keep `time` for now — actually `timed` replaces it entirely. Remove `time` import too.

- [ ] **Step 2: Replace temp file and timing patterns**

Replace the button handler body (lines 34-116). The current structure is:

```python
if st.button("Annotate", type="primary", disabled=not uploaded_file):
    assert uploaded_file is not None
    tmp_path: str | None = None
    try:
        with st.spinner("..."):
            with tempfile.NamedTemporaryFile(...) as tmp_file:
                ...
            start = time.perf_counter_ns()
            doc = convert(tmp_path, converter=converter())
            duration_s = (time.perf_counter_ns() - start) / 1e9
        ...
    except ConversionError as e:
        st.error(str(e))
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
```

Replace with:

```python
if st.button("Annotate", type="primary", disabled=not uploaded_file):
    assert uploaded_file is not None
    try:
        with temp_upload(uploaded_file) as tmp_path:
            with st.spinner(
                "Extracting content... This may take a few minutes for large documents."
            ):
                with timed() as t:
                    doc = convert(tmp_path, converter=converter())

            st.success("Done.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Pictures", len(doc.pictures))
            col2.metric("Tables", len(doc.tables))
            col3.metric("Duration (s)", f"{t.duration_s:.2f}")

            output = build_output(doc, t.duration_s)

            st.download_button(
                label="Download JSON",
                data=json.dumps(output, indent=2),
                file_name=f"{uploaded_file.name}_annotations.json",
                mime="application/json",
            )

            try:
                count = index_elements(
                    cast(list[dict], output["elements"]),
                    uploaded_file.name,
                    embedding_model(),
                    collection(),
                )
                if count > 0:
                    st.info(f"Indexed {count} elements for search.")
                else:
                    st.info("No indexable content found (no descriptions or tables).")
            except Exception:
                st.warning(
                    "Indexing for search failed, but extraction completed successfully."
                )

            for idx, pic in enumerate(doc.pictures, 1):
                with st.expander(f"Picture {idx}", expanded=idx == 1):
                    col_img, col_desc = st.columns(2)
                    image = pic.get_image(doc)
                    if image:
                        col_img.image(image)
                    caption = pic.caption_text(doc=doc)
                    if caption:
                        col_img.caption(caption)
                    desc = get_description(pic)
                    if desc:
                        col_desc.markdown(desc["text"])
                    else:
                        col_desc.write("No description available.")

            for idx, table in enumerate(doc.tables, 1):
                with st.expander(
                    f"Table {idx}",
                    expanded=len(doc.pictures) == 0 and idx == 1,
                ):
                    col_img, col_data = st.columns(2)
                    image = table.get_image(doc)
                    if image:
                        col_img.image(image)
                    caption = table.caption_text(doc=doc)
                    if caption:
                        col_img.caption(caption)
                    df = table.export_to_dataframe(doc=doc)
                    if not df.empty:
                        col_data.dataframe(df)
                    else:
                        col_data.write("Empty table.")

    except ConversionError as e:
        st.error(str(e))
```

- [ ] **Step 3: Run linting to verify**

Run: `uv run ruff check streamlit_app.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: use temp_upload and timed in streamlit_app.py"
```

---

### Task 10: Migrate `pages/segmentation.py` to use `create_granite_vision_model`

**Files:**
- Modify: `pages/segmentation.py`

- [ ] **Step 1: Update import**

Replace:
```python
from pipeline import create_granite_model, create_sam_model, draw_mask, segment
```

With:
```python
from pipeline import create_granite_vision_model, create_sam_model, draw_mask, segment
```

- [ ] **Step 2: Update cache line**

Replace:
```python
granite_model = st.cache_resource(create_granite_model)
```

With:
```python
granite_model = st.cache_resource(create_granite_vision_model)
```

- [ ] **Step 3: Run linting to verify**

Run: `uv run ruff check pages/segmentation.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add pages/segmentation.py
git commit -m "refactor: use create_granite_vision_model in segmentation page"
```

---

### Task 11: Migrate `pages/doctags.py` to use `temp_upload` and `timed`

**Files:**
- Modify: `pages/doctags.py`

- [ ] **Step 1: Update imports**

Replace:
```python
import tempfile
import time
from pathlib import Path
```

Remove all three. Add `timed` and `temp_upload` to the pipeline import:

```python
from pipeline import (
    create_doctags_model,
    export_markdown,
    generate_doctags,
    parse_doctags,
    render_pdf_pages,
    temp_upload,
    timed,
)
```

- [ ] **Step 2: Update PDF processing path**

Replace the PDF branch (lines 34-107) with:

```python
    if is_pdf:
        with temp_upload(uploaded_file) as tmp_path:
            with st.spinner("Rendering PDF pages..."):
                page_images = render_pdf_pages(tmp_path)

            num_pages = len(page_images)
            progress = st.progress(0, text="Generating doctags...")

            all_doctags: list[str] = []
            all_markdown: list[str] = []
            all_docs: list[DoclingDocument | None] = []

            with timed() as t:
                for i, page_image in enumerate(page_images):
                    progress.progress(
                        (i + 1) / num_pages,
                        text=f"Processing page {i + 1} of {num_pages}...",
                    )
                    raw = generate_doctags(page_image, processor, model)
                    all_doctags.append(raw)

                    doc = parse_doctags(raw, page_image) if raw else None
                    all_docs.append(doc)
                    all_markdown.append(export_markdown(doc) if doc else "")

            progress.empty()

            col1, col2 = st.columns(2)
            col1.metric("Pages", num_pages)
            col2.metric("Duration (s)", f"{t.duration_s:.2f}")

            combined_doctags = "\n\n".join(all_doctags)
            combined_markdown = "\n\n---\n\n".join(md for md in all_markdown if md)

            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(
                label="Download all doctags",
                data=combined_doctags,
                file_name=f"{uploaded_file.name}_doctags.txt",
                mime="text/plain",
            )
            dl_col2.download_button(
                label="Download all Markdown",
                data=combined_markdown,
                file_name=f"{uploaded_file.name}_doctags.md",
                mime="text/markdown",
            )

            for i, page_image in enumerate(page_images):
                with st.expander(f"Page {i + 1}", expanded=i == 0):
                    col_img, col_output = st.columns(2)
                    col_img.image(page_image, caption=f"Page {i + 1}")

                    if all_doctags[i]:
                        col_output.code(all_doctags[i], language="xml")

                        if all_docs[i] is not None:
                            col_output.markdown("**Markdown output:**")
                            col_output.markdown(all_markdown[i])
                        else:
                            col_output.warning(
                                "Could not parse doctags into structured document."
                            )
                    else:
                        col_output.warning("Model produced no output for this page.")
```

- [ ] **Step 3: Update single-image processing path**

Replace lines 109-146 with:

```python
    else:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Generating doctags... This may take a few minutes."):
            with timed() as t:
                raw_doctags = generate_doctags(image, processor, model)

        st.metric("Duration (s)", f"{t.duration_s:.2f}")

        col_img, col_output = st.columns(2)
        col_img.image(image, caption="Original")

        if raw_doctags:
            col_output.code(raw_doctags, language="xml")

            doc = parse_doctags(raw_doctags, image)
            if doc is not None:
                md = export_markdown(doc)
                col_output.markdown("**Markdown output:**")
                col_output.markdown(md)
                col_output.download_button(
                    label="Download Markdown",
                    data=md,
                    file_name="doctags_output.md",
                    mime="text/markdown",
                )
            else:
                col_output.warning("Could not parse doctags into structured document.")

            col_output.download_button(
                label="Download raw doctags",
                data=raw_doctags,
                file_name="doctags_output.txt",
                mime="text/plain",
            )
        else:
            col_output.warning("Model produced no output.")
```

- [ ] **Step 4: Run linting to verify**

Run: `uv run ruff check pages/doctags.py`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add pages/doctags.py
git commit -m "refactor: use temp_upload and timed in doctags page"
```

---

### Task 12: Migrate `pages/qa.py` to use shared helpers

**Files:**
- Modify: `pages/qa.py`

- [ ] **Step 1: Update imports and model cache**

Replace:
```python
import tempfile
import time
from pathlib import Path

import pypdfium2
import streamlit as st
from PIL import Image

from pipeline import create_qa_model, generate_qa_response, render_pdf_pages
```

With:
```python
import streamlit as st
from PIL import Image

from pipeline import (
    create_granite_vision_model,
    generate_qa_response,
    get_pdf_page_count,
    render_pdf_pages,
    temp_upload,
    timed,
)
```

Replace:
```python
qa_model = st.cache_resource(create_qa_model)
```

With:
```python
qa_model = st.cache_resource(create_granite_vision_model)
```

- [ ] **Step 2: Replace upload handling and button handler (lines 25-103)**

The key constraint: the page count is needed BEFORE the button click (to show the multiselect), but `temp_upload` creates and destroys a temp file. Use `temp_upload` to get the page count, then `seek(0)` the uploaded file so `temp_upload` can re-read it in the button handler.

```python
page_images: list[Image.Image] = []
is_pdf = False
selected: list[int] = []
valid_upload = True

if uploaded_files:
    pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]

    if pdf_files and len(uploaded_files) > len(pdf_files):
        st.error("Please upload either a single PDF or image files, not both.")
        valid_upload = False
    elif len(pdf_files) > 1:
        st.error("Please upload only one PDF at a time.")
        valid_upload = False
    elif len(pdf_files) == 1:
        is_pdf = True
        with temp_upload(pdf_files[0]) as path:
            total_pages = get_pdf_page_count(path)
        pdf_files[0].seek(0)  # reset for re-read in button handler

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

has_input = valid_upload and bool(uploaded_files) and bool(question)
if is_pdf:
    has_input = has_input and bool(selected)

if st.button("Answer", type="primary", disabled=not has_input):
    assert uploaded_files is not None
    processor, model = qa_model()

    if is_pdf:
        pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
        with temp_upload(pdf_files[0]) as tmp_path:
            with st.spinner("Rendering selected pages..."):
                page_images = render_pdf_pages(
                    tmp_path, page_indices=[i - 1 for i in selected]
                )

            with st.spinner("Generating answer..."):
                with timed() as t:
                    answer = generate_qa_response(
                        page_images, question, processor, model
                    )
    else:
        page_images = [Image.open(f).convert("RGB") for f in uploaded_files]

        with st.spinner("Generating answer..."):
            with timed() as t:
                answer = generate_qa_response(
                    page_images, question, processor, model
                )

    if not answer:
        st.warning("Model produced no output.")
    else:
        col_thumbs, col_answer = st.columns([1, 2])
        with col_thumbs:
            for i, img in enumerate(page_images, 1):
                st.image(img, caption=f"Page {i}", use_container_width=True)
        with col_answer:
            st.markdown(answer)

        st.metric("Duration (s)", f"{t.duration_s:.2f}")

    st.caption("Answers are limited to ~1024 tokens and may be truncated.")
```

- [ ] **Step 3: Run linting to verify**

Run: `uv run ruff check pages/qa.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add pages/qa.py
git commit -m "refactor: use shared helpers in QA page"
```

---

### Task 13: Migrate `pages/search.py` to use `create_granite_vision_model`

**Files:**
- Modify: `pages/search.py`

- [ ] **Step 1: Update import**

Replace:
```python
from pipeline import (
    clear_collection,
    create_embedding_model,
    create_qa_model,
    generate_answer,
    get_collection,
    query_index,
)
```

With:
```python
from pipeline import (
    clear_collection,
    create_embedding_model,
    create_granite_vision_model,
    generate_answer,
    get_collection,
    query_index,
)
```

- [ ] **Step 2: Update cache line**

Replace:
```python
qa_model = st.cache_resource(create_qa_model)
```

With:
```python
qa_model = st.cache_resource(create_granite_vision_model)
```

- [ ] **Step 3: Run linting to verify**

Run: `uv run ruff check pages/search.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add pages/search.py
git commit -m "refactor: use create_granite_vision_model in search page"
```

---

### Task 14: Update `pipeline/__init__.py` re-exports

**Files:**
- Modify: `pipeline/__init__.py`

- [ ] **Step 1: Update re-exports**

Now that all consumers have been updated, replace the entire contents of `pipeline/__init__.py` with:

```python
from pipeline.config import convert, create_converter
from pipeline.doctags import (
    export_markdown,
    generate_doctags,
    get_pdf_page_count,
    parse_doctags,
    render_pdf_pages,
)
from pipeline.models import (
    create_doctags_model,
    create_granite_vision_model,
    generate_response,
)
from pipeline.output import build_output, get_description, get_table_content
from pipeline.qa import generate_qa_response, resize_for_qa
from pipeline.search import (
    clear_collection,
    create_embedding_model,
    generate_answer,
    get_collection,
    index_elements,
    query_index,
)
from pipeline.segmentation import (
    create_sam_model,
    draw_mask,
    segment,
)
from pipeline.utils import timed, temp_upload

__all__ = [
    "build_output",
    "clear_collection",
    "convert",
    "create_converter",
    "create_doctags_model",
    "create_embedding_model",
    "create_granite_vision_model",
    "create_sam_model",
    "draw_mask",
    "export_markdown",
    "generate_answer",
    "generate_doctags",
    "generate_qa_response",
    "generate_response",
    "get_collection",
    "get_description",
    "get_pdf_page_count",
    "get_table_content",
    "index_elements",
    "parse_doctags",
    "query_index",
    "render_pdf_pages",
    "resize_for_qa",
    "segment",
    "temp_upload",
    "timed",
]
```

- [ ] **Step 2: Run all tests to verify nothing is broken**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add pipeline/__init__.py
git commit -m "refactor: update __init__.py re-exports for new modules"
```

---

### Task 15: Final verification

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 2: Run linting**

Run: `uv run ruff check .`
Expected: No errors

- [ ] **Step 3: Run formatter**

Run: `uv run ruff format .`
Expected: No files reformatted (or minor formatting)

- [ ] **Step 4: Run type checker**

Run: `uv run ty check .`
Expected: No new errors introduced

- [ ] **Step 5: Verify no unused imports or dead code**

Check that `pipeline/segmentation.py` no longer exports `create_granite_model`, `pipeline/qa.py` no longer exports `create_qa_model`, and `pipeline/doctags.py` no longer exports `create_doctags_model`.

Run: `uv run ruff check . --select F401`
Expected: No unused import errors

- [ ] **Step 6: Update CLAUDE.md**

Update the Architecture section in `CLAUDE.md` to reflect the new file structure:
- Add `pipeline/models.py` — `_load_vision_model`, `create_granite_vision_model`, `create_doctags_model`, `generate_response`
- Add `pipeline/utils.py` — `temp_upload`, `timed`
- Update `pipeline/segmentation.py` — remove `create_granite_model` reference
- Update `pipeline/qa.py` — remove `create_qa_model` reference, note `generate_qa_response` delegates to `generate_response`
- Update `pipeline/search.py` — note `generate_answer` delegates to `generate_response`
- Update `pipeline/doctags.py` — remove `create_doctags_model` reference, add `get_pdf_page_count`
- Update `pipeline/__init__.py` re-exports list
- Add `tests/test_models.py` and `tests/test_utils.py`

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for refactored module structure"
```
