# Refactoring Improvements Design

## Goal

Reduce code duplication and improve memory efficiency across the pipeline and UI layers without changing external behavior.

## Changes

### 1. Shared model loader (`pipeline/models.py`)

Extract a `_load_vision_model(repo_id, device)` helper that handles device detection and `AutoProcessor`/`AutoModelForVision2Seq` loading. The three current factories (`create_granite_model`, `create_qa_model`, `create_doctags_model`) become thin wrappers calling this helper.

Additionally, since `create_granite_model` and `create_qa_model` load the same `granite-vision-3.3-2b` weights, unify them into a single `create_granite_vision_model()` factory. The segmentation page, QA page, search page, and `streamlit_app.py` all share one cached instance. `create_doctags_model` remains separate (different repo: `granite-docling-258M`). `create_sam_model` stays in `segmentation.py` since SAM uses different classes (`SamModel`/`SamProcessor`).

**Impact:** Up to ~4GB memory saved when multiple pages are visited. Three near-identical functions replaced by one.

### 2. Shared generate helper (`pipeline/models.py`)

Extract `generate_response(conversation, processor, model, max_new_tokens, images)` that encapsulates the repeated pattern:
1. `apply_chat_template` with `tokenize=True, return_dict=True, return_tensors="pt"`
2. Move inputs to device
3. `model.generate` under `torch.inference_mode`
4. Trim input tokens from output
5. Decode with `skip_special_tokens=True`

This replaces duplicate code in `qa.py:generate_qa_response`, `search.py:generate_answer`, and `segmentation.py:segment`.

The `doctags.py:generate_doctags` variant is slightly different (uses `processor(text=..., images=...)` instead of `apply_chat_template` with `tokenize=True`, and decodes with `skip_special_tokens=False`), so it stays as-is.

### 3. Shared temp file context manager (`pipeline/utils.py`)

```python
@contextmanager
def temp_upload(uploaded_file, suffix=".pdf"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        path = f.name
    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)
```

Replaces the try/finally pattern in `streamlit_app.py`, `pages/doctags.py`, and `pages/qa.py`.

### 4. Shared timer context manager (`pipeline/utils.py`)

```python
@contextmanager
def timed():
    start = time.perf_counter_ns()
    result = type("Timer", (), {"duration_s": 0.0})()
    try:
        yield result
    finally:
        result.duration_s = (time.perf_counter_ns() - start) / 1e9
```

Replaces the 4 instances of `start = time.perf_counter_ns()` / `duration_s = (... - start) / 1e9`.

### 5. Optimize token counting in `_chunk_text`

Replace per-part `tokenizer.encode()` calls with a character-based length estimate (`len(text) // 4`) for the initial over/under check, only using exact tokenization near chunk boundaries. This reduces tokenizer calls from O(n) to O(chunks).

### 6. Avoid redundant PDF open in QA page

In `pages/qa.py`, the PDF is opened once via `pypdfium2.PdfDocument` to get page count, then again via `render_pdf_pages()`. Change to pass the page count from the first open rather than opening twice, or use `render_pdf_pages` to also return the total page count.

## File changes

| File | Change |
|---|---|
| `pipeline/models.py` | **New.** `_load_vision_model`, `create_granite_vision_model`, `create_doctags_model`, `generate_response` |
| `pipeline/utils.py` | **New.** `temp_upload`, `timed` |
| `pipeline/segmentation.py` | Remove `create_granite_model`. Import `create_granite_vision_model` from `models`. Replace inline generate logic with `generate_response`. Keep `create_sam_model` here. |
| `pipeline/qa.py` | Remove `create_qa_model`, `generate_qa_response` body logic. Import from `models`. `generate_qa_response` calls `generate_response`. |
| `pipeline/search.py` | Remove inline generate logic from `generate_answer`. Import `generate_response` from `models`. Optimize `_chunk_text`. |
| `pipeline/doctags.py` | Remove `create_doctags_model`. Import from `models`. Keep `generate_doctags` as-is (different generation pattern). |
| `pipeline/__init__.py` | Update re-exports: replace `create_granite_model` + `create_qa_model` with `create_granite_vision_model`. Add `create_doctags_model`. Add `temp_upload`, `timed`. |
| `streamlit_app.py` | Use `temp_upload`, `timed`, `create_granite_vision_model` (shared). |
| `pages/segmentation.py` | Use `create_granite_vision_model` instead of `create_granite_model`. |
| `pages/doctags.py` | Use `temp_upload`, `timed`. |
| `pages/qa.py` | Use `temp_upload`, `timed`, `create_granite_vision_model`. Remove redundant PDF open. |
| `pages/search.py` | Use `create_granite_vision_model` instead of `create_qa_model`. |
| `tests/test_segmentation.py` | No changes (tests internal helpers, not model loading). |
| `tests/test_qa.py` | Update mock paths for `create_qa_model` → `create_granite_vision_model`. |
| `tests/test_doctags.py` | Update mock paths for `create_doctags_model`. |
| `tests/test_search.py` | No changes (already mocks at call sites). |
| `tests/test_models.py` | **New.** Tests for `_load_vision_model`, `create_granite_vision_model`, `create_doctags_model`, `generate_response`. |
| `tests/test_utils.py` | **New.** Tests for `temp_upload`, `timed`. |

## Backward compatibility

- `create_granite_model` and `create_qa_model` are removed from the public API. They were only used by UI pages (not external consumers).
- All pipeline functions keep the same signatures and return types.
- Test fixtures and test PDF remain unchanged.

## What stays unchanged

- `pipeline/config.py` — Docling converter is unrelated to these refactors.
- `pipeline/output.py` — No duplication to address.
- `create_sam_model` — Uses different model classes, stays in `segmentation.py`.
- `generate_doctags` — Different generation pattern (no `apply_chat_template` with `tokenize=True`), stays as-is.
