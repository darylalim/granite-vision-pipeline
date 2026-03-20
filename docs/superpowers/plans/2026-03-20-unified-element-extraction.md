# Unified Element Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Broaden the pipeline from picture-only descriptions to structured extraction of tables alongside pictures, using a unified document-element model.

**Architecture:** Replace the `pictures` array in the output with a unified `elements` array where each entry has a `type` discriminator (`"picture"` or `"table"`) and type-specific `content`. Add `get_table_content()` and `build_element()` to the pipeline. Update the Streamlit UI to render tables with `st.dataframe` and pictures as before.

**Tech Stack:** Docling (`TableItem`, `TableData`, `TableCell`, `export_to_dataframe`, `export_to_markdown`), Streamlit (`st.dataframe`), pandas

---

## File Map

- **Modify:** `pipeline/output.py` — add `get_table_content()`, `build_element()`, refactor `build_output()`
- **Modify:** `pipeline/__init__.py` — re-export `get_table_content`
- **Modify:** `pipeline/config.py` — add `generate_table_images = True`
- **Modify:** `tests/test_output.py` — update existing tests, add table tests
- **Modify:** `streamlit_app.py` — render tables with `st.dataframe`, 3-column metrics, unified element loop
- **Modify:** `CLAUDE.md` — update architecture docs

---

### Task 1: Add `get_table_content()` with TDD

**Files:**
- Modify: `tests/test_output.py`
- Modify: `pipeline/output.py`

- [ ] **Step 1: Add table test helpers and imports to test file**

Add these imports and helpers at the top of `tests/test_output.py`, after the existing imports and helpers:

```python
from docling_core.types.doc.document import (
    DescriptionAnnotation,
    DescriptionMetaField,
    DoclingDocument,
    PictureItem,
    PictureMeta,
    TableCell,
    TableData,
    TableItem,
)

from pipeline import build_output, get_description, get_table_content
```

Add this helper after `_make_picture_with_annotation`:

```python
def _make_table(
    index: int,
    cells: list[TableCell] | None = None,
    num_rows: int = 0,
    num_cols: int = 0,
) -> TableItem:
    """Create a TableItem with the given cells."""
    data = TableData(
        table_cells=cells or [],
        num_rows=num_rows,
        num_cols=num_cols,
    )
    return TableItem(
        self_ref=f"#/tables/{index}",
        data=data,
    )
```

Update `_make_doc` to accept tables:

```python
def _make_doc(
    pictures: list[PictureItem] | None = None,
    tables: list[TableItem] | None = None,
) -> DoclingDocument:
    """Create a DoclingDocument with the given pictures and tables."""
    doc = DoclingDocument(name="test")
    if pictures:
        doc.pictures = pictures
    if tables:
        doc.tables = tables
    return doc
```

- [ ] **Step 2: Write failing tests for `get_table_content`**

Add these tests after the existing `get_description` tests at the end of `tests/test_output.py`:

```python
# --- Tests for get_table_content ---


def test_get_table_content_with_headers() -> None:
    cells = [
        TableCell(text="Name", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1, column_header=True),
        TableCell(text="Age", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2, column_header=True),
        TableCell(text="Alice", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=1),
        TableCell(text="30", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=1, end_col_offset_idx=2),
    ]
    table = _make_table(0, cells=cells, num_rows=2, num_cols=2)
    doc = _make_doc(tables=[table])
    result = get_table_content(table, doc)
    assert isinstance(result["markdown"], str)
    assert result["data"]["columns"] == ["Name", "Age"]
    assert result["data"]["rows"] == [["Alice", "30"]]


def test_get_table_content_without_headers() -> None:
    cells = [
        TableCell(text="a", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1),
        TableCell(text="b", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2),
    ]
    table = _make_table(0, cells=cells, num_rows=1, num_cols=2)
    doc = _make_doc(tables=[table])
    result = get_table_content(table, doc)
    assert result["data"]["columns"] == ["0", "1"]
    assert result["data"]["rows"] == [["a", "b"]]


def test_get_table_content_empty_table() -> None:
    table = _make_table(0)
    doc = _make_doc(tables=[table])
    result = get_table_content(table, doc)
    assert result["data"] == {"columns": [], "rows": []}
    assert isinstance(result["markdown"], str)


def test_get_table_content_merged_columns() -> None:
    cells = [
        TableCell(text="H1", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1, column_header=True),
        TableCell(text="H2", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2, column_header=True),
        TableCell(text="wide", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=2, col_span=2),
        TableCell(text="wide", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=2, col_span=2),
    ]
    table = _make_table(0, cells=cells, num_rows=2, num_cols=2)
    doc = _make_doc(tables=[table])
    result = get_table_content(table, doc)
    assert result["data"]["columns"] == ["H1", "H2"]
    assert len(result["data"]["rows"]) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_output.py -k "get_table_content" -v`
Expected: FAIL (ImportError — `get_table_content` does not exist yet)

- [ ] **Step 4: Implement `get_table_content` in `pipeline/output.py`**

Add `TableItem` to the imports and add the function after `get_description`:

```python
from docling_core.types.doc.document import (
    DescriptionAnnotation,
    DoclingDocument,
    PictureItem,
    TableItem,
)


def get_table_content(table: TableItem, doc: DoclingDocument) -> dict[str, object]:
    """Extract table content as markdown and structured data."""
    df = table.export_to_dataframe(doc=doc)
    return {
        "markdown": table.export_to_markdown(doc=doc),
        "data": {
            "columns": [str(c) for c in df.columns],
            "rows": df.values.tolist(),
        },
    }
```

- [ ] **Step 5: Re-export `get_table_content` from `pipeline/__init__.py`**

Update `pipeline/__init__.py`:

```python
from pipeline.config import convert, create_converter
from pipeline.output import build_output, get_description, get_table_content

__all__ = [
    "build_output",
    "convert",
    "create_converter",
    "get_description",
    "get_table_content",
]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_output.py -k "get_table_content" -v`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest -v`
Expected: All existing tests still pass, 4 new tests pass.

- [ ] **Step 8: Commit**

```bash
git add pipeline/output.py pipeline/__init__.py tests/test_output.py
git commit -m "feat: add get_table_content for structured table extraction"
```

---

### Task 2: Add `build_element()` and refactor `build_output()` with TDD

**Files:**
- Modify: `tests/test_output.py`
- Modify: `pipeline/output.py`

- [ ] **Step 1: Write failing tests for `build_element`**

Add a top-level import for `build_element` in `tests/test_output.py` alongside the existing imports:

```python
from pipeline.output import build_element
```

Then add these tests after the `get_table_content` tests:

```python
# --- Tests for build_element ---


def test_build_element_picture() -> None:
    pic = _make_picture(0, text="A chart.", created_by="test-model")
    doc = _make_doc([pic])
    result = build_element(pic, doc, element_number=1, element_type="picture")
    assert result == {
        "element_number": 1,
        "type": "picture",
        "reference": "#/pictures/0",
        "caption": "",
        "content": {"description": {"text": "A chart.", "created_by": "test-model"}},
    }


def test_build_element_table() -> None:
    cells = [
        TableCell(text="X", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1, column_header=True),
        TableCell(text="1", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=1),
    ]
    table = _make_table(0, cells=cells, num_rows=2, num_cols=1)
    doc = _make_doc(tables=[table])
    result = build_element(table, doc, element_number=2, element_type="table")
    assert result["element_number"] == 2
    assert result["type"] == "table"
    assert result["reference"] == "#/tables/0"
    assert result["content"]["data"]["columns"] == ["X"]
    assert result["content"]["data"]["rows"] == [["1"]]
    assert isinstance(result["content"]["markdown"], str)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_output.py::test_build_element_picture tests/test_output.py::test_build_element_table -v`
Expected: FAIL (ImportError — `build_element` does not exist yet)

- [ ] **Step 3: Implement `build_element` in `pipeline/output.py`**

Add `from typing import Literal` to the top of `pipeline/output.py` with the other imports. Then add this function after `get_table_content`:

```python
def build_element(
    item: PictureItem | TableItem,
    doc: DoclingDocument,
    element_number: int,
    element_type: Literal["picture", "table"],
) -> dict[str, object]:
    """Build a unified element dict for a picture or table."""
    if element_type == "picture":
        assert isinstance(item, PictureItem)
        content: dict[str, object] = {"description": get_description(item)}
    else:
        assert isinstance(item, TableItem)
        content = get_table_content(item, doc)
    return {
        "element_number": element_number,
        "type": element_type,
        "reference": item.self_ref,
        "caption": item.caption_text(doc=doc) or "",
        "content": content,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_output.py::test_build_element_picture tests/test_output.py::test_build_element_table -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for refactored `build_output`**

Update the existing `build_output` tests in `tests/test_output.py` to use the new `elements` structure. Replace all existing `build_output` tests (from `test_top_level_keys` through `test_meta_description_preferred_over_annotations`) with:

```python
# --- Tests for build_output (unified elements) ---


def test_top_level_keys() -> None:
    doc = _make_doc()
    result = build_output(doc, 1.5)
    assert set(result.keys()) == {"document_info", "elements"}


def test_document_info_fields() -> None:
    doc = _make_doc([_make_picture(0)], [_make_table(0)])
    result = build_output(doc, 2.34)
    info = result["document_info"]
    assert isinstance(info, dict)
    assert info["num_pictures"] == 1
    assert info["num_tables"] == 1
    assert info["total_duration_s"] == 2.34


def test_picture_element_structure() -> None:
    pic = _make_picture(0, text="A test description.", created_by="test-model")
    doc = _make_doc([pic])
    result = build_output(doc, 0.5)
    elements = result["elements"]
    assert isinstance(elements, list)
    assert len(elements) == 1
    entry = elements[0]
    assert entry["element_number"] == 1
    assert entry["type"] == "picture"
    assert entry["reference"] == "#/pictures/0"
    assert entry["caption"] == ""
    assert entry["content"]["description"]["text"] == "A test description."
    assert entry["content"]["description"]["created_by"] == "test-model"


def test_multiple_pictures_in_elements() -> None:
    pics = [
        _make_picture(0, text="First pic.", created_by="model-a"),
        _make_picture(1, text="Second pic.", created_by="model-b"),
    ]
    doc = _make_doc(pics)
    result = build_output(doc, 3.0)
    elements = result["elements"]
    assert len(elements) == 2
    assert elements[0]["element_number"] == 1
    assert elements[0]["content"]["description"]["text"] == "First pic."
    assert elements[1]["element_number"] == 2
    assert elements[1]["content"]["description"]["text"] == "Second pic."


def test_table_element_structure() -> None:
    cells = [
        TableCell(text="Col", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1, column_header=True),
        TableCell(text="val", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=1),
    ]
    table = _make_table(0, cells=cells, num_rows=2, num_cols=1)
    doc = _make_doc(tables=[table])
    result = build_output(doc, 0.5)
    elements = result["elements"]
    assert len(elements) == 1
    entry = elements[0]
    assert entry["element_number"] == 1
    assert entry["type"] == "table"
    assert entry["content"]["data"]["columns"] == ["Col"]


def test_mixed_pictures_and_tables() -> None:
    pics = [
        _make_picture(0, text="Pic 1.", created_by="model"),
        _make_picture(1, text="Pic 2.", created_by="model"),
    ]
    cells = [
        TableCell(text="A", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1),
    ]
    tables = [_make_table(0, cells=cells, num_rows=1, num_cols=1)]
    doc = _make_doc(pics, tables)
    result = build_output(doc, 3.0)
    elements = result["elements"]
    assert len(elements) == 3
    assert elements[0]["type"] == "picture"
    assert elements[0]["element_number"] == 1
    assert elements[1]["type"] == "picture"
    assert elements[1]["element_number"] == 2
    assert elements[2]["type"] == "table"
    assert elements[2]["element_number"] == 3


def test_no_description_in_picture_element() -> None:
    pic = _make_picture(0)
    doc = _make_doc([pic])
    result = build_output(doc, 0.5)
    assert result["elements"][0]["content"]["description"] is None


def test_empty_document() -> None:
    doc = _make_doc()
    result = build_output(doc, 0.0)
    assert result["document_info"]["num_pictures"] == 0
    assert result["document_info"]["num_tables"] == 0
    assert result["elements"] == []


def test_description_from_annotations_fallback_in_element() -> None:
    pic = _make_picture_with_annotation(0, text="A chart.", provenance="test-model")
    doc = _make_doc([pic])
    result = build_output(doc, 1.0)
    description = result["elements"][0]["content"]["description"]
    assert description is not None
    assert description["text"] == "A chart."
    assert description["created_by"] == "test-model"


def test_meta_description_preferred_over_annotations_in_element() -> None:
    pic = _make_picture(0, text="From meta.", created_by="meta-model")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        pic.annotations.append(
            DescriptionAnnotation(text="From annotations.", provenance="ann-model")
        )
    doc = _make_doc([pic])
    result = build_output(doc, 1.0)
    description = result["elements"][0]["content"]["description"]
    assert description["text"] == "From meta."
    assert description["created_by"] == "meta-model"
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_output.py::test_top_level_keys tests/test_output.py::test_document_info_fields tests/test_output.py::test_mixed_pictures_and_tables -v`
Expected: FAIL (output still has `pictures` key, not `elements`)

- [ ] **Step 7: Refactor `build_output` in `pipeline/output.py`**

Replace the existing `build_output` function:

```python
def build_output(doc: DoclingDocument, duration_s: float) -> dict[str, object]:
    """Build the output dictionary from a converted document."""
    elements: list[dict[str, object]] = []
    counter = 1
    for pic in doc.pictures:
        elements.append(build_element(pic, doc, counter, "picture"))
        counter += 1
    for table in doc.tables:
        elements.append(build_element(table, doc, counter, "table"))
        counter += 1
    return {
        "document_info": {
            "num_pictures": len(doc.pictures),
            "num_tables": len(doc.tables),
            "total_duration_s": duration_s,
        },
        "elements": elements,
    }
```

- [ ] **Step 8: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 9: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean.

- [ ] **Step 10: Commit**

```bash
git add pipeline/output.py tests/test_output.py
git commit -m "feat: add build_element and refactor build_output to unified elements"
```

---

### Task 3: Enable table image generation in config

**Files:**
- Modify: `pipeline/config.py`

- [ ] **Step 1: Add `generate_table_images = True` to `create_converter`**

In `pipeline/config.py`, add after line 39 (`pipeline_options.generate_picture_images = True`):

```python
    pipeline_options.generate_table_images = True
```

- [ ] **Step 2: Run existing config tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: All pass (existing tests only check `isinstance`).

- [ ] **Step 3: Commit**

```bash
git add pipeline/config.py
git commit -m "feat: enable table image generation in converter config"
```

---

### Task 4: Update Streamlit UI for unified elements

**Files:**
- Modify: `streamlit_app.py`

- [ ] **Step 1: Verify imports**

No import changes needed in `streamlit_app.py`. The existing import line already has everything the UI needs:

```python
from pipeline import build_output, convert, create_converter, get_description
```

`get_table_content` is not needed in the UI — the dataframe is obtained directly from `table.export_to_dataframe(doc=doc)` and the JSON download uses `build_output()` which calls `get_table_content` internally.

- [ ] **Step 2: Update spinner text**

Replace line 22-23:

```python
        with st.spinner(
            "Describing pictures... This may take a few minutes for large documents."
        ):
```

With:

```python
        with st.spinner(
            "Extracting content... This may take a few minutes for large documents."
        ):
```

- [ ] **Step 3: Update metrics row to 3 columns**

Replace lines 35-37:

```python
        col1, col2 = st.columns(2)
        col1.metric("Pictures", len(doc.pictures))
        col2.metric("Duration (s)", f"{duration_s:.2f}")
```

With:

```python
        col1, col2, col3 = st.columns(3)
        col1.metric("Pictures", len(doc.pictures))
        col2.metric("Tables", len(doc.tables))
        col3.metric("Duration (s)", f"{duration_s:.2f}")
```

- [ ] **Step 4: Add table expanders after picture expanders**

After the existing picture expander loop (after line 59), add:

```python
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
```

- [ ] **Step 5: Add pandas import for `export_to_dataframe` usage**

No extra import needed — `export_to_dataframe` is called on the `TableItem` object which handles the pandas import internally.

- [ ] **Step 6: Run the app manually**

Run: `uv run streamlit run streamlit_app.py`

Verify:
- Spinner says "Extracting content..."
- 3-column metrics row shows Pictures, Tables, Duration
- Picture expanders still work as before
- Table expanders appear after pictures with image on left and interactive dataframe on right
- JSON download contains unified `elements` array

- [ ] **Step 7: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean.

- [ ] **Step 8: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: update UI for unified element display with table support"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Architecture section**

Replace the `pipeline/output.py` line:

```
- `pipeline/output.py` — `build_output()` pure function that builds output dict from a `DoclingDocument` and duration; `get_description()` reads from `pic.meta.description` with fallback to `pic.annotations`
```

With:

```
- `pipeline/output.py` — `build_output()` produces a unified `elements` array from pictures and tables via `build_element()`; `get_description()` extracts picture descriptions from `meta` with fallback to `annotations`; `get_table_content()` extracts table markdown and structured column/row data
```

- [ ] **Step 2: Update key details**

Replace the output JSON line:

```
- Output JSON contains `document_info` (count, timing) and a `pictures` array (reference, caption, description)
```

With:

```
- Output JSON contains `document_info` (picture count, table count, timing) and an `elements` array with `type` discriminator (`"picture"` or `"table"`) and type-specific `content`
```

Replace the upload flow line:

```
- Upload flow: upload PDF, click "Annotate", spinner, metrics (picture count, duration), JSON download, and per-picture preview in expanders (image + description)
```

With:

```
- Upload flow: upload PDF, click "Annotate", spinner, metrics (picture count, table count, duration), JSON download, per-picture preview in expanders (image + description), and per-table preview in expanders (image + interactive dataframe)
```

- [ ] **Step 3: Update `pipeline/__init__.py` line**

Replace:

```
- `pipeline/__init__.py` — re-exports public API (`convert`, `create_converter`, `build_output`, `get_description`)
```

With:

```
- `pipeline/__init__.py` — re-exports public API (`convert`, `create_converter`, `build_output`, `get_description`, `get_table_content`)
```

- [ ] **Step 4: Update Tests section**

Replace:

```
- `tests/test_output.py` — `build_output()` and `get_description()` with real Docling objects, annotations fallback, meta priority over annotations
```

With:

```
- `tests/test_output.py` — `build_output()`, `build_element()`, `get_description()`, and `get_table_content()` with real Docling objects; covers pictures, tables, mixed documents, annotations fallback, meta priority
```

- [ ] **Step 5: Run full test suite one final time**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 6: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean.

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for unified element extraction"
```
