# UI Improvements Design

## Overview

Make the Granite Vision Pipeline UI simpler and friendlier by restructuring the app layout, adding shared UI helpers, and improving consistency across all 5 pages.

## File Structure Changes

### Current
```
streamlit_app.py          # PDF extraction + landing page
pages/
  segmentation.py
  doctags.py
  qa.py
  search.py
```

### Proposed
```
streamlit_app.py          # Navigation hub with st.navigation() routing
streamlit_home.py         # Landing/home page content
ui_helpers.py             # Shared UI patterns
pages/
  extraction.py           # PDF extraction (moved from streamlit_app.py)
  segmentation.py
  doctags.py
  qa.py
  search.py
examples/
  sample.jpg              # Sample image for segmentation/doctags demo
  sample.pdf              # Sample PDF (copied from test fixtures)
```

- `pages/__init__.py` is removed — no longer needed since `st.navigation()` replaces Streamlit's legacy auto-discovery of the `pages/` directory

- `streamlit_app.py` becomes the navigation hub using `st.navigation()` / `st.Page()` API, which controls both sidebar labels and routing
- PDF extraction logic moves to `pages/extraction.py`
- `ui_helpers.py` provides shared UI functions
- Test PDF `tests/data/pdf/test_pictures.pdf` is copied to `examples/sample.pdf` to decouple demos from test fixtures
- New `examples/sample.jpg` for image-based pages — a general-purpose photo with distinct objects (e.g., a desk with objects) suitable for segmentation prompts like "the cup" or "the keyboard"
- No changes to `pipeline/` — all changes are UI-only

## Page Routing and Naming

Use `st.navigation()` and `st.Page()` in `streamlit_app.py` to define all pages with explicit titles. This controls sidebar labels and browser tab titles from one place, decoupling display names from filenames.

```python
home = st.Page("streamlit_home.py", title="Home", icon=":material/home:")
extraction = st.Page("pages/extraction.py", title="PDF Extraction")
segmentation = st.Page("pages/segmentation.py", title="Image Segmentation")
doctags = st.Page("pages/doctags.py", title="Document Parsing")
qa = st.Page("pages/qa.py", title="Multipage QA")
search = st.Page("pages/search.py", title="Document Search")

st.set_page_config(page_title="Granite Vision Pipeline")
pg = st.navigation([home, extraction, segmentation, doctags, qa, search])
pg.run()
```

Note: With `st.navigation()`, `st.set_page_config()` is called once in the main file. Subpages must NOT call it. The home page content lives in a separate file (`streamlit_home.py`) referenced by `st.Page()`.

### Page Link Paths

Landing page cards use `st.page_link()` with the `st.Page` objects:
```python
st.page_link(extraction, label="PDF Extraction")
```

## Landing Page (`streamlit_home.py`)

- Title: "Granite Vision Pipeline"
- Tagline: "Document AI powered by IBM Granite models"
- 5 cards in a grid (3 columns top row, 2 columns bottom row), each with:
  - Capability name
  - One-line description
  - `st.page_link()` to navigate
- Cards use `st.container(border=True)`
- No model loading — page loads instantly

## Shared UI Helpers (`ui_helpers.py`)

### `show_upload_preview(uploaded_files, max_height=200)`
Shows thumbnail preview of uploaded file(s) immediately after upload. Accepts a single file or a list of files. For images, displays the image thumbnail. For PDFs, shows filename and file size only (no page count — that would require pipeline imports). The caller can optionally display page count separately if they already have it. When given multiple files (QA page), shows thumbnails in a horizontal row using `st.columns`, capped at 8.

### `show_help(title, supported_formats, description, model_info)`
Renders a collapsed `st.expander("How this works")` with consistent formatting: supported formats, what to expect, model used, output format. Each page calls with page-specific content.

### `show_metrics_bar(metrics: dict)`
Takes a dict like `{"Pictures": 5, "Duration (s)": "2.34"}` and renders them in equal `st.columns`.

### `load_example(file_path) -> BytesIO`
Returns a `BytesIO` wrapper with `.name` and `.size` attributes so it can substitute for a Streamlit `UploadedFile`. The `.size` is set to `len(buf.getvalue())`. Powers "Try with example" buttons.

### `show_sidebar_status(models: dict[str, bool], index_count: int | None = None)`
Shows model loaded/not-loaded status and optional search index count in the sidebar. Takes a dict like `{"Granite Vision": True, "SAM": False}` where booleans are computed by the caller using `st.session_state` flags (see Sidebar Status section).

No `elapsed_spinner` helper — Streamlit's single-threaded execution model prevents updating UI while blocking code runs. Pages use the existing pattern: `st.spinner()` for feedback during processing, `st.metric()` for duration after completion.

### Pipeline import policy
`ui_helpers.py` has no pipeline imports. Any data that requires pipeline calls (e.g., PDF page count) is computed at the call site and passed as a parameter.

## "Try with Example" Interaction Model

Since `st.file_uploader` cannot be programmatically populated, the example button works via `st.session_state`:

1. "Try with example" button and file uploader are shown side by side in two columns
2. Clicking "Try with example" sets `st.session_state["use_example_{page}"] = True` and triggers `st.rerun()`
3. Page logic checks: if `uploaded_file` exists, use it; else if `st.session_state.get("use_example_{page}")`, call `load_example()` to get the example file
4. When the user uploads their own file, the example flag is cleared
5. A small `st.caption("Using example file")` is shown when demo mode is active

This means the action button and downstream logic receive the same file-like object regardless of source.

## Per-Page Improvements

### Consistent Page Structure (all pages)

1. Title + one-liner
2. `show_help()` expander
3. File uploader + "Try with example" button side by side
4. `show_upload_preview()` immediately after upload
5. Action button
6. `show_metrics_bar()` for results
7. Results content
8. Download buttons

### PDF Extraction (`pages/extraction.py`)

- Moved from `streamlit_app.py`, same logic
- Defines its own `st.cache_resource` wrappers for `converter`, `embedding_model`, and `collection` (same pattern as current `streamlit_app.py`). `search.py` independently defines its own wrappers — both hit the same underlying cache since `st.cache_resource` is keyed by the wrapped function.
- The existing `st.set_page_config()` call is removed (now in `streamlit_app.py` only)
- Add help expander, upload preview
- Add "Try with example" using `examples/sample.pdf`

### Image Segmentation (`pages/segmentation.py`)

- Title controlled by `st.Page()` in main app — page itself just uses `st.title("Image Segmentation")`
- Add help expander with tip about prompt specificity
- Friendlier error: "Couldn't find that object. Try a more specific description, like 'the red car on the left' instead of 'car'."
- Download filename includes source: `{uploaded_file.name}_mask.png`
- Add "Try with example" using `examples/sample.jpg` with a default prompt matching the image content

### Document Parsing (`pages/doctags.py`)

- Title: "Document Parsing" (controlled by `st.Page()`, page uses `st.title("Document Parsing")`)
- Add help expander explaining doctags format
- Add upload preview
- Two "Try with example" buttons: "Try with example image" and "Try with example PDF". Each sets its own session state flag and loads the corresponding file. Only one example can be active at a time — selecting one clears the other.

### Multipage QA (`pages/qa.py`)

- Add help expander
- Add upload preview (thumbnails of selected pages)
- Fix page numbering: thread `selected` list (1-indexed page numbers chosen by user) through to the display loop. When rendering result thumbnails, use `selected[i]` instead of `i+1` for the caption. For image uploads (non-PDF), synthesize `selected = list(range(1, len(page_images) + 1))` so the display loop uses one unified path.

### Document Search (`pages/search.py`)

- Add help expander explaining workflow (extract first, then search)
- Add index summary table showing filenames and element counts. Implementation: call `collection.get(include=["metadatas"])`, group by `metadata["source"]`, count per source. Shown in an expander "Indexed documents" to avoid clutter. For large collections this is a full scan — acceptable since this is a local tool, not a production service.
- Minimal other changes — already well-structured

## Sidebar Status

Each page calls `show_sidebar_status()` with its known model states and optionally the search index count.

**Model status tracking**: Pages set `st.session_state` flags after the model is actually called (not at definition time, since `st.cache_resource` is lazy). For example, in `pages/extraction.py`:
```python
converter = st.cache_resource(create_converter)

# Later, inside the button handler after actually calling the model:
doc = convert(tmp_path, converter=converter())
st.session_state["model_docling"] = True
```

Pages pass their known flags to `show_sidebar_status()`. Models not yet called show as "Not loaded", which is accurate — the model weights are not in memory until first invocation. On subsequent reruns, `st.session_state` persists so the flag remains `True`.

**Search index count**: Only `pages/extraction.py` and `pages/search.py` pass `index_count` since they already load the collection. Other pages pass `None` and the count is omitted.

### Per-page model mapping

| Page | Models reported |
|------|----------------|
| PDF Extraction | Docling, Embedding |
| Image Segmentation | Granite Vision, SAM |
| Document Parsing | Docling |
| Multipage QA | Granite Vision |
| Document Search | Granite Vision, Embedding |

Display format in sidebar:
```
--- Models ---
Granite Vision: Loaded
SAM: Not loaded
Docling: Not loaded
Embedding: Not loaded

--- Search Index ---
3 documents indexed
```

## Out of Scope

- No changes to `pipeline/` module
- No changes to tests (UI-only changes, no testable pipeline logic added)
- No new dependencies
- `st.set_page_config()` is only called in `streamlit_app.py`; subpages must not call it
