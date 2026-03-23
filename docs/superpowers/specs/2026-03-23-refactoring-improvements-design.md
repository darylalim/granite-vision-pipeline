# Refactoring Improvements Design

## Goal

Reduce code duplication and improve memory efficiency across the pipeline and UI layers without changing external behavior.

## Changes

### 1. Shared model loader (`pipeline/models.py`)

Extract a `_load_vision_model(repo_id, device)` helper that handles device detection and `AutoProcessor`/`AutoModelForVision2Seq` loading. The three current factories (`create_granite_model`, `create_qa_model`, `create_doctags_model`) become thin wrappers calling this helper.

Since `create_granite_model` and `create_qa_model` load the same `granite-vision-3.3-2b` weights, unify them into a single `create_granite_vision_model()` factory. `create_doctags_model` uses a different repo (`granite-docling-258M`) so it stays as a separate factory, but moves to `models.py` to use the shared `_load_vision_model` helper. `create_sam_model` stays in `segmentation.py` since SAM uses different classes (`SamModel`/`SamProcessor`).

**Model sharing mechanism:** To share one model instance across Streamlit pages, define one cached loader in a shared module (e.g., a `_cache.py` or at the top of each page importing the same function). Streamlit's `st.cache_resource` caches by function identity — if all pages call `st.cache_resource(create_granite_vision_model)` where `create_granite_vision_model` is the same imported function object, they share the cache. This is already how `create_qa_model` is shared between `pages/qa.py` and `pages/search.py`.

**Impact:** Memory saved when the segmentation page and QA/search pages are visited in the same session (one `granite-vision-3.3-2b` instance instead of two). Three near-identical factory functions replaced by one helper.

### 2. Shared generate helper (`pipeline/models.py`)

Extract `generate_response(conversation, processor, model, max_new_tokens)` that encapsulates the repeated pattern:
1. `apply_chat_template` with `tokenize=True, return_dict=True, return_tensors="pt"`
2. Move inputs to device (inferred from `model.parameters()`)
3. `model.generate` under `torch.inference_mode`
4. Trim input tokens from output
5. Decode with `skip_special_tokens=True`

This replaces duplicate code in `qa.py:generate_qa_response`, `search.py:generate_answer`, and `segmentation.py:segment`.

**Note on `segment()`:** Currently `segment()` decodes the full output without trimming input tokens. The shared helper always trims. This is a safe behavioral change because `segment()` then parses the decoded text with `re.search(r"<seg>.*</seg>")`, which finds the tags regardless of whether the input prompt is prepended. Trimming just produces cleaner decoded text.

**Note on `doctags.py`:** `generate_doctags` uses a different generation flow (`processor(text=..., images=...)` instead of `apply_chat_template` with `tokenize=True`, and decodes with `skip_special_tokens=False`), so it stays as-is.

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

Replaces the 4 instances of `start = time.perf_counter_ns()` / `duration_s = (... - start) / 1e9`. In `pages/doctags.py`, there are two separate timing blocks (one for multi-page PDF processing, one for single-image processing). The `timed()` context manager applies to both — the PDF path wraps the entire processing loop (including progress updates), which is what the current code already times.

### 5. Optimize token counting in `_chunk_text`

Replace per-part `tokenizer.encode()` calls with a character-based length estimate (`len(text) // 4`) for the initial over/under check, only using exact tokenization near chunk boundaries. This reduces tokenizer calls from O(n) to O(chunks).

Note: This is a minor optimization since `_chunk_text` is only invoked for elements exceeding 8K tokens (rare). Including it for completeness but it has the lowest priority.

### 6. Avoid redundant PDF open in QA page

In `pages/qa.py`, the PDF is opened once via `pypdfium2.PdfDocument` to get page count, then again inside `render_pdf_pages()`. Add an optional `total_pages` return to `render_pdf_pages`, or extract a `get_pdf_page_count(path)` helper in `pipeline/doctags.py` that `pages/qa.py` can use without importing `pypdfium2` directly.

## File changes

| File | Change |
|---|---|
| `pipeline/models.py` | **New.** `_load_vision_model`, `create_granite_vision_model`, `create_doctags_model`, `generate_response` |
| `pipeline/utils.py` | **New.** `temp_upload`, `timed` |
| `pipeline/segmentation.py` | Remove `create_granite_model`. Import `create_granite_vision_model` from `models`. Replace inline generate logic in `segment()` with `generate_response`. Keep `create_sam_model` here. |
| `pipeline/qa.py` | Remove `create_qa_model`. Import `create_granite_vision_model` from `models`. `generate_qa_response` becomes a thin wrapper: builds conversation, calls `generate_response`. Keep `resize_for_qa` as-is. |
| `pipeline/search.py` | Remove inline generate logic from `generate_answer`, call `generate_response`. Optimize `_chunk_text`. |
| `pipeline/doctags.py` | Remove `create_doctags_model`. Import from `models`. Add `get_pdf_page_count()` helper. Keep `generate_doctags` as-is (different generation pattern). |
| `pipeline/__init__.py` | Update re-exports: replace `create_granite_model` + `create_qa_model` with `create_granite_vision_model`. Update `create_doctags_model` import source (already exported, just changing where it comes from). Add `temp_upload`, `timed`, `get_pdf_page_count`, `generate_response`. |
| `streamlit_app.py` | Use `temp_upload` and `timed`. (Does not use vision models — no change there.) |
| `pages/segmentation.py` | Use `create_granite_vision_model` instead of `create_granite_model`. |
| `pages/doctags.py` | Use `temp_upload`, `timed`. |
| `pages/qa.py` | Use `temp_upload`, `timed`, `create_granite_vision_model` instead of `create_qa_model`. Use `get_pdf_page_count` instead of direct `pypdfium2` import. |
| `pages/search.py` | Use `create_granite_vision_model` instead of `create_qa_model`. |
| `tests/test_segmentation.py` | No changes (tests internal helpers, not model loading). |
| `tests/test_qa.py` | Update mock paths for model loading. `generate_qa_response` signature unchanged, so test logic stays the same. |
| `tests/test_doctags.py` | Update mock paths for `create_doctags_model`. |
| `tests/test_search.py` | Update `generate_answer` tests to mock `generate_response` instead of inline model calls. |
| `tests/test_models.py` | **New.** Tests for `_load_vision_model`, `create_granite_vision_model`, `create_doctags_model`, `generate_response`. |
| `tests/test_utils.py` | **New.** Tests for `temp_upload`, `timed`. |

## Backward compatibility

- `create_granite_model` and `create_qa_model` are removed from the public API. They were only used by UI pages (not external consumers).
- `generate_qa_response` and `generate_answer` keep their existing signatures — they become thin wrappers around `generate_response`.
- `create_doctags_model` stays in the public API, just moves from `pipeline.doctags` to `pipeline.models`.
- All pipeline functions keep the same signatures and return types.
- Test fixtures and test PDF remain unchanged.

## What stays unchanged

- `pipeline/config.py` — Docling converter is unrelated to these refactors.
- `pipeline/output.py` — No duplication to address.
- `create_sam_model` — Uses different model classes, stays in `segmentation.py`.
- `generate_doctags` — Different generation pattern, stays as-is in `doctags.py`.
