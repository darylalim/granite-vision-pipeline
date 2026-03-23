# UI Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the Streamlit UI with a landing page, shared helpers, consistent page layout, upload previews, help expanders, example files, better error messages, and sidebar status indicators.

**Architecture:** Replace the current implicit multipage routing with `st.navigation()` / `st.Page()`. Extract PDF extraction from `streamlit_app.py` into `pages/extraction.py`. Create `ui_helpers.py` for shared UI patterns (preview, help, metrics, examples, sidebar). Add `streamlit_home.py` as a card-based landing page. All changes are UI-only — `pipeline/` is untouched.

**Tech Stack:** Streamlit (`st.navigation`, `st.Page`, `st.session_state`, `st.cache_resource`), PIL for image previews, `io.BytesIO` for example file loading.

**Spec:** `docs/superpowers/specs/2026-03-23-ui-improvements-design.md`

---

### Task 1: Create `ui_helpers.py`

**Files:**
- Create: `ui_helpers.py`

- [ ] **Step 1: Create `ui_helpers.py` with all shared helper functions**

```python
"""Shared UI helper functions for Streamlit pages.

No pipeline imports — any data requiring pipeline calls is computed
at the call site and passed as a parameter.
"""

import io
from pathlib import Path

import streamlit as st
from PIL import Image


def show_upload_preview(
    uploaded_files: object | list[object],
    max_height: int = 200,
) -> None:
    """Show thumbnail preview of uploaded file(s).

    For images, displays the image thumbnail. For PDFs, shows filename
    and file size. Accepts a single file or a list of files.
    """
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    files = [f for f in uploaded_files if f is not None]
    if not files:
        return

    cols = st.columns(min(len(files), 8))
    for i, f in enumerate(files[:8]):
        name = getattr(f, "name", "file")
        with cols[i]:
            if name.lower().endswith(".pdf"):
                size = getattr(f, "size", None)
                if size is not None:
                    st.caption(f"**{name}**\n{size / 1024:.0f} KB")
                else:
                    st.caption(f"**{name}**")
            else:
                try:
                    img = Image.open(f)
                    st.image(img, caption=name, width=max_height)
                    f.seek(0)
                except Exception:
                    st.caption(f"**{name}**")


def show_help(
    supported_formats: str,
    description: str,
    model_info: str,
) -> None:
    """Render a collapsed 'How this works' expander with consistent formatting."""
    with st.expander("How this works"):
        st.markdown(f"**Supported formats:** {supported_formats}")
        st.markdown(description)
        st.markdown(f"**Model:** {model_info}")


def show_metrics_bar(metrics: dict[str, object]) -> None:
    """Render metrics in equal columns."""
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


class _ExampleFile(io.BytesIO):
    """BytesIO wrapper with .name and .size attributes for UploadedFile compat."""

    name: str
    size: int


def load_example(file_path: str) -> _ExampleFile:
    """Load a file as a BytesIO wrapper that substitutes for st.UploadedFile."""
    data = Path(file_path).read_bytes()
    buf = _ExampleFile(data)
    buf.name = Path(file_path).name
    buf.size = len(data)
    return buf


def show_sidebar_status(
    models: dict[str, bool],
    index_count: int | None = None,
) -> None:
    """Show model status and optional search index count in the sidebar."""
    with st.sidebar:
        st.markdown("**Models**")
        for name, loaded in models.items():
            status = "Loaded" if loaded else "Not loaded"
            st.text(f"{name}: {status}")

        if index_count is not None:
            st.markdown("**Search Index**")
            st.text(
                f"{index_count} document{'s' if index_count != 1 else ''} indexed"
            )
```

- [ ] **Step 2: Verify the file was created correctly**

Run: `uv run ruff check ui_helpers.py && uv run ruff format --check ui_helpers.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add ui_helpers.py
git commit -m "feat(ui): add shared UI helper functions"
```

---

### Task 2: Create example files

**Files:**
- Create: `examples/sample.pdf` (copied from test fixture)
- Create: `examples/sample.jpg` (generated placeholder)

- [ ] **Step 1: Create `examples/` directory and copy sample PDF**

```bash
mkdir -p examples
cp tests/data/pdf/test_pictures.pdf examples/sample.pdf
```

- [ ] **Step 2: Create a sample JPG image**

Generate a simple image with distinct colored shapes suitable for segmentation prompts. Use Python/PIL:

```python
from PIL import Image, ImageDraw

img = Image.new("RGB", (640, 480), (245, 245, 245))
draw = ImageDraw.Draw(img)
# Red rectangle (left)
draw.rectangle([50, 150, 200, 350], fill=(220, 50, 50))
# Blue circle (center)
draw.ellipse([250, 150, 400, 300], fill=(50, 50, 220))
# Green triangle (right)
draw.polygon([(500, 350), (550, 150), (600, 350)], fill=(50, 180, 50))
img.save("examples/sample.jpg", quality=90)
```

Run this as a one-shot script:
```bash
uv run python -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (640, 480), (245, 245, 245))
draw = ImageDraw.Draw(img)
draw.rectangle([50, 150, 200, 350], fill=(220, 50, 50))
draw.ellipse([250, 150, 400, 300], fill=(50, 50, 220))
draw.polygon([(500, 350), (550, 150), (600, 350)], fill=(50, 180, 50))
img.save('examples/sample.jpg', quality=90)
"
```

- [ ] **Step 3: Verify files exist**

```bash
ls -la examples/
```

Expected: `sample.pdf` and `sample.jpg` both present

- [ ] **Step 4: Commit**

```bash
git add examples/
git commit -m "feat(ui): add example files for demo mode"
```

---

### Task 3: Create `pages/extraction.py`

Move PDF extraction logic from `streamlit_app.py` to `pages/extraction.py`, adding UI improvements.

**Files:**
- Create: `pages/extraction.py`

- [ ] **Step 1: Create `pages/extraction.py`**

This is the current `streamlit_app.py` logic with these changes:
- Remove `st.set_page_config()` (now in main app only)
- Add `show_help()`, `show_upload_preview()`, `show_metrics_bar()`, `show_sidebar_status()`
- Add "Try with example" via `st.session_state`
- Set session state flags after model calls

```python
import json
from typing import cast

import streamlit as st
from docling.exceptions import ConversionError
from docling_core.types.doc.document import DoclingDocument

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
from ui_helpers import (
    load_example,
    show_help,
    show_metrics_bar,
    show_sidebar_status,
    show_upload_preview,
)

converter = st.cache_resource(create_converter)
embedding_model = st.cache_resource(create_embedding_model)
collection = st.cache_resource(get_collection)

EXAMPLE_PDF = "examples/sample.pdf"

st.title("PDF Extraction")
st.write(
    "Extract and describe pictures and tables in PDF documents using IBM Granite Vision."
)

show_help(
    supported_formats="PDF",
    description=(
        "Uploads a PDF and uses Docling to extract pictures and tables. "
        "Pictures are described using Granite Vision. Tables are parsed into "
        "structured data. Results are available as a JSON download and are "
        "automatically indexed for the Document Search page."
    ),
    model_info="[granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) via Docling",
)

col_upload, col_example = st.columns([3, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload file", type=["pdf"])
with col_example:
    st.markdown("")  # spacing
    if st.button("Try with example"):
        st.session_state["use_example_extraction"] = True
        st.rerun()

# Resolve file: user upload takes priority over example
active_file = uploaded_file
if uploaded_file:
    st.session_state.pop("use_example_extraction", None)
elif st.session_state.get("use_example_extraction"):
    active_file = load_example(EXAMPLE_PDF)
    st.caption("Using example file")

if active_file:
    show_upload_preview(active_file)


def _render_elements(doc: DoclingDocument) -> None:
    """Display extracted pictures and tables in expanders."""
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


if st.button("Annotate", type="primary", disabled=not active_file):
    assert active_file is not None
    file_name = getattr(active_file, "name", "document.pdf")
    try:
        with temp_upload(active_file) as tmp_path:
            with st.spinner(
                "Extracting content... This may take a few minutes for large documents."
            ):
                with timed() as t:
                    doc = convert(tmp_path, converter=converter())
                    st.session_state["model_docling"] = True

        st.success("Done.")

        show_metrics_bar({
            "Pictures": len(doc.pictures),
            "Tables": len(doc.tables),
            "Duration (s)": f"{t.duration_s:.2f}",
        })

        output = build_output(doc, t.duration_s)

        st.download_button(
            label="Download JSON",
            data=json.dumps(output, indent=2),
            file_name=f"{file_name}_annotations.json",
            mime="application/json",
        )

        try:
            count = index_elements(
                cast(list[dict], output["elements"]),
                file_name,
                embedding_model(),
                collection(),
            )
            st.session_state["model_embedding"] = True
            if count > 0:
                st.info(f"Indexed {count} elements for search.")
            else:
                st.info("No indexable content found (no descriptions or tables).")
        except Exception:
            st.warning(
                "Indexing for search failed, but extraction completed successfully."
            )

        _render_elements(doc)

    except ConversionError as e:
        st.error(str(e))

# Sidebar status
coll = collection()
show_sidebar_status(
    models={
        "Docling": st.session_state.get("model_docling", False),
        "Embedding": st.session_state.get("model_embedding", False),
    },
    index_count=coll.count(),
)
```

- [ ] **Step 2: Verify lint passes**

Run: `uv run ruff check pages/extraction.py && uv run ruff format --check pages/extraction.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add pages/extraction.py
git commit -m "feat(ui): create extraction page with UI improvements"
```

---

### Task 4: Create `streamlit_home.py` and rewrite `streamlit_app.py`

**Files:**
- Create: `streamlit_home.py`
- Modify: `streamlit_app.py` (full rewrite)
- Delete: `pages/__init__.py`

- [ ] **Step 1: Create `streamlit_home.py` with navigation cards**

```python
import streamlit as st

st.title("Granite Vision Pipeline")
st.write("Document AI powered by IBM Granite models.")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("PDF Extraction")
        st.write("Extract and describe pictures and tables from PDF documents.")
        st.page_link("pages/extraction.py", label="Open", icon=":material/picture_as_pdf:")

with col2:
    with st.container(border=True):
        st.subheader("Image Segmentation")
        st.write("Segment objects in images using natural language prompts.")
        st.page_link("pages/segmentation.py", label="Open", icon=":material/content_cut:")

with col3:
    with st.container(border=True):
        st.subheader("Document Parsing")
        st.write("Parse document images to structured text in doctags format.")
        st.page_link("pages/doctags.py", label="Open", icon=":material/code:")

col4, col5 = st.columns(2)

with col4:
    with st.container(border=True):
        st.subheader("Multipage QA")
        st.write("Ask questions about document pages using vision AI.")
        st.page_link("pages/qa.py", label="Open", icon=":material/question_answer:")

with col5:
    with st.container(border=True):
        st.subheader("Document Search")
        st.write("Search across extracted content with RAG-powered answers.")
        st.page_link("pages/search.py", label="Open", icon=":material/search:")
```

- [ ] **Step 2: Rewrite `streamlit_app.py` as navigation hub**

Replace the entire contents of `streamlit_app.py` with:

```python
import streamlit as st

st.set_page_config(page_title="Granite Vision Pipeline")

pg = st.navigation([
    st.Page("streamlit_home.py", title="Home", icon=":material/home:"),
    st.Page("pages/extraction.py", title="PDF Extraction"),
    st.Page("pages/segmentation.py", title="Image Segmentation"),
    st.Page("pages/doctags.py", title="Document Parsing"),
    st.Page("pages/qa.py", title="Multipage QA"),
    st.Page("pages/search.py", title="Document Search"),
])
pg.run()
```

- [ ] **Step 3: Remove `pages/__init__.py`**

```bash
rm pages/__init__.py
```

No longer needed — `st.navigation()` replaces Streamlit's legacy auto-discovery.

- [ ] **Step 4: Verify lint passes**

Run: `uv run ruff check streamlit_app.py streamlit_home.py && uv run ruff format --check streamlit_app.py streamlit_home.py`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py streamlit_home.py
git rm pages/__init__.py
git commit -m "feat(ui): add landing page and st.navigation() routing"
```

---

### Task 5: Update `pages/segmentation.py`

**Files:**
- Modify: `pages/segmentation.py`

- [ ] **Step 1: Rewrite `pages/segmentation.py` with UI improvements**

Changes from current file:
- Title: "Image Segmentation" (drop "(Experimental)")
- Add `show_help()` with prompt specificity tip
- Add "Try with example" with default prompt
- Add `show_upload_preview()`
- Friendlier error message
- Download filename includes source name
- Add `show_sidebar_status()`

```python
import io

import streamlit as st
from PIL import Image

from pipeline import create_granite_vision_model, create_sam_model, draw_mask, segment
from ui_helpers import (
    load_example,
    show_help,
    show_sidebar_status,
    show_upload_preview,
)

granite_model = st.cache_resource(create_granite_vision_model)
sam_model = st.cache_resource(create_sam_model)

EXAMPLE_IMAGE = "examples/sample.jpg"

st.title("Image Segmentation")
st.write(
    "Segment objects in images using natural language prompts. "
    "Powered by Granite Vision with SAM refinement."
)

show_help(
    supported_formats="PNG, JPG, JPEG",
    description=(
        "Upload an image and describe the object you want to segment. "
        "The model identifies the object and produces a binary mask. "
        "For best results, use specific descriptions like "
        "'the red car on the left' instead of just 'car'."
    ),
    model_info=(
        "[granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) "
        "+ [SAM (sam-vit-huge)](https://huggingface.co/facebook/sam-vit-huge)"
    ),
)

col_upload, col_example = st.columns([3, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
with col_example:
    st.markdown("")  # spacing
    if st.button("Try with example"):
        st.session_state["use_example_segmentation"] = True
        st.rerun()

# Resolve file: user upload takes priority over example
active_file = uploaded_file
if uploaded_file:
    st.session_state.pop("use_example_segmentation", None)
elif st.session_state.get("use_example_segmentation"):
    active_file = load_example(EXAMPLE_IMAGE)
    st.caption("Using example file")

if active_file:
    show_upload_preview(active_file)

default_prompt = "the red rectangle" if st.session_state.get("use_example_segmentation") else ""
prompt = st.text_input(
    "Segmentation prompt",
    value=default_prompt,
    placeholder="e.g., the dog on the left",
)

if st.button("Segment", type="primary", disabled=not active_file or not prompt):
    assert active_file is not None
    file_name = getattr(active_file, "name", "image")
    image = Image.open(active_file)

    with st.spinner("Running segmentation... This may take a few minutes."):
        mask = segment(
            image,
            prompt,
            granite=granite_model(),
            sam=sam_model(),
        )
        st.session_state["model_granite_vision"] = True
        st.session_state["model_sam"] = True

    if mask is None:
        st.error(
            "Couldn't find that object. Try a more specific description, "
            "like 'the red car on the left' instead of 'car'."
        )
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
            file_name=f"{file_name}_mask.png",
            mime="image/png",
        )

show_sidebar_status(
    models={
        "Granite Vision": st.session_state.get("model_granite_vision", False),
        "SAM": st.session_state.get("model_sam", False),
    },
)
```

- [ ] **Step 2: Verify lint passes**

Run: `uv run ruff check pages/segmentation.py && uv run ruff format --check pages/segmentation.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add pages/segmentation.py
git commit -m "feat(ui): improve segmentation page with helpers and examples"
```

---

### Task 6: Update `pages/doctags.py`

**Files:**
- Modify: `pages/doctags.py`

- [ ] **Step 1: Rewrite `pages/doctags.py` with UI improvements**

Changes from current file:
- Title: "Document Parsing" (was "DocTags Generation (Experimental)")
- Add `show_help()` explaining doctags format
- Two "Try with example" buttons (image and PDF)
- Add `show_upload_preview()`
- Add `show_metrics_bar()` for single-image mode
- Add `show_sidebar_status()`

```python
import streamlit as st
from docling_core.types.doc.document import DoclingDocument
from PIL import Image

from pipeline import (
    create_doctags_model,
    export_markdown,
    generate_doctags,
    parse_doctags,
    render_pdf_pages,
    temp_upload,
    timed,
)
from ui_helpers import (
    load_example,
    show_help,
    show_metrics_bar,
    show_sidebar_status,
    show_upload_preview,
)

doctags_model = st.cache_resource(create_doctags_model)

EXAMPLE_IMAGE = "examples/sample.jpg"
EXAMPLE_PDF = "examples/sample.pdf"

st.title("Document Parsing")
st.write(
    "Parse document images to structured text in doctags format. "
    "Powered by IBM Granite Docling."
)

show_help(
    supported_formats="PNG, JPG, JPEG, PDF",
    description=(
        "Converts document images or PDF pages into doctags — a structured XML "
        "format that captures text, layout, and document structure. The doctags "
        "output can be further converted to Markdown. For PDFs, each page is "
        "processed independently."
    ),
    model_info="[granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M)",
)

col_upload, col_ex_img, col_ex_pdf = st.columns([3, 1, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload file", type=["png", "jpg", "jpeg", "pdf"])
with col_ex_img:
    st.markdown("")  # spacing
    if st.button("Example image"):
        st.session_state["use_example_doctags"] = "image"
        st.rerun()
with col_ex_pdf:
    st.markdown("")  # spacing
    if st.button("Example PDF"):
        st.session_state["use_example_doctags"] = "pdf"
        st.rerun()

# Resolve file: user upload takes priority over example
active_file = uploaded_file
if uploaded_file:
    st.session_state.pop("use_example_doctags", None)
elif st.session_state.get("use_example_doctags") == "image":
    active_file = load_example(EXAMPLE_IMAGE)
    st.caption("Using example image")
elif st.session_state.get("use_example_doctags") == "pdf":
    active_file = load_example(EXAMPLE_PDF)
    st.caption("Using example PDF")

if active_file:
    show_upload_preview(active_file)

is_pdf = active_file is not None and getattr(active_file, "name", "").lower().endswith(
    ".pdf"
)

if st.button("Generate", type="primary", disabled=not active_file):
    assert active_file is not None
    file_name = getattr(active_file, "name", "document")
    processor, model = doctags_model()
    st.session_state["model_docling"] = True

    if is_pdf:
        with temp_upload(active_file) as tmp_path:
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

            show_metrics_bar({
                "Pages": num_pages,
                "Duration (s)": f"{t.duration_s:.2f}",
            })

            combined_doctags = "\n\n".join(all_doctags)
            combined_markdown = "\n\n---\n\n".join(md for md in all_markdown if md)

            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(
                label="Download all doctags",
                data=combined_doctags,
                file_name=f"{file_name}_doctags.txt",
                mime="text/plain",
            )
            dl_col2.download_button(
                label="Download all Markdown",
                data=combined_markdown,
                file_name=f"{file_name}_doctags.md",
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

    else:
        image = Image.open(active_file).convert("RGB")

        with st.spinner("Generating doctags... This may take a few minutes."):
            with timed() as t:
                raw_doctags = generate_doctags(image, processor, model)

        show_metrics_bar({"Duration (s)": f"{t.duration_s:.2f}"})

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
                    file_name=f"{file_name}_doctags.md",
                    mime="text/markdown",
                )
            else:
                col_output.warning("Could not parse doctags into structured document.")

            col_output.download_button(
                label="Download raw doctags",
                data=raw_doctags,
                file_name=f"{file_name}_doctags.txt",
                mime="text/plain",
            )
        else:
            col_output.warning("Model produced no output.")

show_sidebar_status(
    models={"Docling": st.session_state.get("model_docling", False)},
)
```

- [ ] **Step 2: Verify lint passes**

Run: `uv run ruff check pages/doctags.py && uv run ruff format --check pages/doctags.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add pages/doctags.py
git commit -m "feat(ui): improve doctags page with helpers and examples"
```

---

### Task 7: Update `pages/qa.py`

**Files:**
- Modify: `pages/qa.py`

- [ ] **Step 1: Rewrite `pages/qa.py` with UI improvements**

Changes from current file:
- Title: "Multipage QA" (drop "(Experimental)")
- Add `show_help()`
- Add `show_upload_preview()` after upload
- Add `show_metrics_bar()`
- Fix page numbering: use `selected[i]` for PDF captions, synthesize sequential for images
- Add `show_sidebar_status()`
- Add "Try with example" using the example PDF

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
from ui_helpers import (
    load_example,
    show_help,
    show_metrics_bar,
    show_sidebar_status,
    show_upload_preview,
)

qa_model = st.cache_resource(create_granite_vision_model)

EXAMPLE_PDF = "examples/sample.pdf"

st.title("Multipage QA")
st.write(
    "Ask questions about document pages using IBM Granite Vision. "
    "Upload a PDF or up to 8 images, then type your question."
)

show_help(
    supported_formats="PDF, PNG, JPG, JPEG (up to 8 images)",
    description=(
        "Upload a single PDF or up to 8 images. For PDFs, select which pages "
        "to include (up to 8). Type a question and the model will analyze all "
        "selected pages together to generate an answer. Images are resized to "
        "768px max dimension to fit within memory limits."
    ),
    model_info="[granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b)",
)

col_upload, col_example = st.columns([3, 1])
with col_upload:
    uploaded_files = st.file_uploader(
        "Upload file(s)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
with col_example:
    st.markdown("")  # spacing
    if st.button("Try with example"):
        st.session_state["use_example_qa"] = True
        st.rerun()

# Resolve files: user upload takes priority over example
if uploaded_files:
    st.session_state.pop("use_example_qa", None)
elif st.session_state.get("use_example_qa"):
    uploaded_files = [load_example(EXAMPLE_PDF)]
    st.caption("Using example file")

page_images: list[Image.Image] = []
is_pdf = False
selected: list[int] = []
valid_upload = True

if uploaded_files:
    show_upload_preview(uploaded_files)

    pdf_files = [f for f in uploaded_files if getattr(f, "name", "").lower().endswith(".pdf")]

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
    st.session_state["model_granite_vision"] = True

    if is_pdf:
        pdf_files = [f for f in uploaded_files if getattr(f, "name", "").lower().endswith(".pdf")]
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
        selected = list(range(1, len(page_images) + 1))

        with st.spinner("Generating answer..."):
            with timed() as t:
                answer = generate_qa_response(page_images, question, processor, model)

    if not answer:
        st.warning("Model produced no output.")
    else:
        col_thumbs, col_answer = st.columns([1, 2])
        with col_thumbs:
            for i, img in enumerate(page_images):
                st.image(
                    img,
                    caption=f"Page {selected[i]}",
                    use_container_width=True,
                )
        with col_answer:
            st.markdown(answer)

        show_metrics_bar({"Duration (s)": f"{t.duration_s:.2f}"})

    st.caption("Answers are limited to ~1024 tokens and may be truncated.")

show_sidebar_status(
    models={"Granite Vision": st.session_state.get("model_granite_vision", False)},
)
```

- [ ] **Step 2: Verify lint passes**

Run: `uv run ruff check pages/qa.py && uv run ruff format --check pages/qa.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add pages/qa.py
git commit -m "feat(ui): improve QA page with helpers, examples, and page numbering fix"
```

---

### Task 8: Update `pages/search.py`

**Files:**
- Modify: `pages/search.py`

- [ ] **Step 1: Rewrite `pages/search.py` with UI improvements**

Changes from current file:
- Add `show_help()` explaining workflow
- Add index summary table in expander (filenames + element counts)
- Add `show_sidebar_status()`

```python
from collections import Counter

import streamlit as st

from pipeline import (
    clear_collection,
    create_embedding_model,
    create_granite_vision_model,
    generate_answer,
    get_collection,
    query_index,
)
from ui_helpers import show_help, show_sidebar_status

embedding_model = st.cache_resource(create_embedding_model)
collection = st.cache_resource(get_collection)
qa_model = st.cache_resource(create_granite_vision_model)

st.title("Document Search")
st.write(
    "Search across extracted document content using natural language questions. "
    "Documents are automatically indexed when processed on the PDF Extraction page."
)

show_help(
    supported_formats="N/A (searches previously extracted documents)",
    description=(
        "This page searches across documents that have been processed on the "
        "PDF Extraction page. Extracted pictures and tables are embedded and "
        "stored in a local vector database. Enter a question to find relevant "
        "content and get a RAG-powered answer."
    ),
    model_info=(
        "[granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) "
        "for search, [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) "
        "for answer generation"
    ),
)

coll = collection()


@st.cache_data(ttl=5)
def _get_doc_count() -> int:
    return coll.count()


doc_count = _get_doc_count()
st.metric("Indexed documents", doc_count)

if doc_count == 0:
    st.info("No documents indexed yet. Extract a PDF first.")
else:
    with st.expander("Indexed documents"):
        all_meta = coll.get(include=["metadatas"])
        source_counts: Counter[str] = Counter()
        for meta in all_meta["metadatas"] or []:
            source = meta.get("source", "Unknown") if meta else "Unknown"
            source_counts[source] += 1
        for source, count in sorted(source_counts.items()):
            st.text(f"{source}: {count} elements")

question = st.text_input(
    "Question", placeholder="e.g., What does the revenue chart show?"
)

if st.button("Search", type="primary", disabled=not question or doc_count == 0):
    model = embedding_model()
    st.session_state["model_embedding"] = True

    with st.spinner("Searching..."):
        results = query_index(question, model, coll)

    if not results:
        st.warning("No relevant results found for your question.")
    else:
        processor, gen_model = qa_model()
        st.session_state["model_granite_vision"] = True

        with st.spinner("Generating answer..."):
            answer = generate_answer(question, results, processor, gen_model)

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Sources")
        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            type_label = meta.get("type", "element").capitalize()
            source = meta.get("source", "Unknown")
            elem_num = meta.get("element_number", "?")
            similarity = result["similarity"]

            with st.expander(
                f"{type_label} (Element {elem_num}) from {source} — similarity: {similarity:.2f}",
                expanded=i == 1,
            ):
                st.text(result["text"])

st.divider()

with st.popover("Clear Index", disabled=doc_count == 0):
    st.write(f"This will delete all {doc_count} indexed documents. Are you sure?")
    if st.button("Confirm Clear", type="primary"):
        clear_collection(coll)
        st.success("Index cleared.")
        st.rerun()

show_sidebar_status(
    models={
        "Granite Vision": st.session_state.get("model_granite_vision", False),
        "Embedding": st.session_state.get("model_embedding", False),
    },
    index_count=doc_count,
)
```

- [ ] **Step 2: Verify lint passes**

Run: `uv run ruff check pages/search.py && uv run ruff format --check pages/search.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add pages/search.py
git commit -m "feat(ui): improve search page with help, index summary, and sidebar"
```

---

### Task 9: Final verification

**Files:**
- All modified files

- [ ] **Step 1: Run full lint and format check**

```bash
uv run ruff check . && uv run ruff format --check .
```

Expected: No errors

- [ ] **Step 2: Run type checker**

```bash
uv run ty check .
```

Expected: No new errors (existing warnings acceptable)

- [ ] **Step 3: Run all tests**

```bash
uv run pytest
```

Expected: All tests pass (no pipeline changes, tests should be unaffected)

- [ ] **Step 4: Verify app starts**

```bash
uv run streamlit run streamlit_app.py --server.headless true &
sleep 3
curl -s http://localhost:8501 | head -20
kill %1
```

Expected: HTML response from Streamlit (confirms app starts without import errors)

- [ ] **Step 5: Fix any issues found in steps 1-4**

If lint, type, or test errors arise, fix them and re-run the failing check.

- [ ] **Step 6: Final commit (if fixes were needed)**

```bash
git add -A
git commit -m "fix(ui): resolve lint/type/test issues from UI improvements"
```

---

### Task 10: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md to reflect the new UI structure**

Update the Architecture > UI section to reflect:
- `streamlit_app.py` is now the navigation hub using `st.navigation()` / `st.Page()`
- `streamlit_home.py` is the landing page with navigation cards
- `ui_helpers.py` provides shared UI functions (preview, help, metrics, examples, sidebar)
- `pages/extraction.py` is the PDF extraction page (moved from `streamlit_app.py`)
- Page titles updated: "Image Segmentation" (no experimental), "Document Parsing" (was DocTags Generation), "Multipage QA" (no experimental)
- `examples/` directory with sample files for demo mode
- `pages/__init__.py` removed

Also update any references to `streamlit_app.py` being the PDF extraction page — it is now the navigation entry point only.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for new UI structure"
```
