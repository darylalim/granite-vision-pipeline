# Unified Element Extraction

**Goal:** Broaden the pipeline from picture-only descriptions to structured extraction of tables alongside pictures, using a unified document-element model.

**Motivation:** granite-vision-3.3-2b is designed for visual document understanding across tables, charts, infographics, and diagrams. The current pipeline only extracts picture descriptions. Tables are the most common structured element in PDFs and Docling already provides rich table parsing — this is the natural first expansion.

---

## Output Structure

Replace the `pictures`-only JSON output with a unified `elements` array. Each element has a `type` discriminator and type-specific `content`.

```json
{
  "document_info": {
    "num_pictures": 2,
    "num_tables": 3,
    "total_duration_s": 12.5
  },
  "elements": [
    {
      "element_number": 1,
      "type": "picture",
      "reference": "#/pictures/0",
      "caption": "Figure 1: Revenue growth",
      "content": {
        "description": { "text": "A bar chart showing...", "created_by": "granite-vision-3.3-2b" }
      }
    },
    {
      "element_number": 2,
      "type": "table",
      "reference": "#/tables/0",
      "caption": "Table 1: Q4 Results",
      "content": {
        "markdown": "| Quarter | Revenue |\n|---|---|\n| Q4 | 1.2M |",
        "data": { "columns": ["Quarter", "Revenue"], "rows": [["Q4", "1.2M"]] }
      }
    }
  ]
}
```

**Key decisions:**

- `element_number` is a globally sequential counter across all element types (not per-type). Pictures are numbered first, then tables continue the sequence.
- Elements ordered by type (pictures first, then tables), not document order. Docling exposes them as separate lists; interleaving by page position adds complexity not needed yet.
- `content` is type-specific: pictures get `description`, tables get `markdown` + `data` (columns/rows).
- `document_info` gains `num_tables` alongside `num_pictures`.
- Table `data` uses a simple columns+rows structure derived from `TableItem.export_to_dataframe()`. Column names are always strings — integer indices from headerless tables are converted via `str()`.
- Empty tables produce `{"markdown": "", "data": {"columns": [], "rows": []}}`.
- This is a breaking change to the JSON output (the `pictures` key is removed). Acceptable at v0.1.0 with no known downstream consumers.

---

## Pipeline Module Changes

### `pipeline/output.py`

- `get_description(pic)` — unchanged.
- New `get_table_content(table, doc)` — calls `table.export_to_markdown(doc=doc)` and `table.export_to_dataframe(doc=doc)`, converts the DataFrame to `{"columns": [...], "rows": [[...], ...]}`. Column names are always converted to strings.
- `build_output(doc, duration_s)` — refactored to iterate pictures and tables, calling `build_element()` for each, returning the unified structure.
- New `build_element(item, doc, element_number, element_type)` — produces the common dict shape with type-specific content. Dispatches by `element_type`: `"picture"` calls `get_description(item)`, `"table"` calls `get_table_content(item, doc)`. Both types use `item.caption_text(doc=doc)` for captions (requires `doc`).

### `pipeline/__init__.py`

- Re-export `get_table_content` alongside existing public API.

### `pipeline/config.py`

- Add `pipeline_options.generate_table_images = True` so that `table.get_image(doc)` returns table images for UI display. Docling defaults this to `False`.

---

## Streamlit UI Changes

### `streamlit_app.py`

- Spinner text updated: "Extracting content..." (covers both pictures and tables).
- Metrics row: 3 columns — pictures count, tables count, duration.
- Download button uses the new unified output structure.
- Element display loop: iterate `doc.pictures` then `doc.tables`, each in an expander labeled by type and number (e.g., "Picture 1", "Table 3").
- Picture expanders: unchanged (image left, description right).
- Table expanders: source image on the left (via `table.get_image(doc)` if available), `st.dataframe` on the right for interactive grid.

---

## Test Changes

### `tests/test_output.py`

- Existing `build_output` tests updated to use `elements` array instead of `pictures`.
- `get_description` tests unchanged.
- New `get_table_content` tests: markdown output, columns/rows structure, empty table, table with merged cells.
- New `build_element` tests: picture element shape, table element shape.
- New `build_output` tests for mixed content: documents with both pictures and tables, tables-only, ordering.
- New `_make_table()` helper factory using real `TableItem`/`TableData` objects.

### `tests/test_config.py`

- No changes.

---

## CLAUDE.md Updates

Update the Architecture section:

- `pipeline/output.py` — mention `build_element()`, `get_table_content()`, and the unified `elements` array.
- Output JSON: `elements` array with `type` discriminator replaces `pictures` array.
- `document_info` includes `num_tables`.
- Upload flow description: mention table rendering with `st.dataframe`.
