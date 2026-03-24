# Granite Vision Pipeline

Streamlit web app with five capabilities:

1. **PDF Extraction** — extract and describe pictures and tables in PDF documents
2. **Image Segmentation** — segment objects in images using natural language prompts
3. **Document Parsing** — parse document images and PDFs to structured text in doctags format
4. **Multipage QA** — answer questions across up to 8 document pages
5. **Document Search** — search across extracted content and get RAG-powered answers

Powered by [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b), [granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2), [SAM](https://huggingface.co/facebook/sam-vit-huge), and [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M). Navigate between features using the sidebar. Each page includes a "Try with example" button for quick demos.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Features

**PDF Extraction** — Upload a PDF to extract pictures with AI-generated descriptions and tables with structured data. Results available as JSON download with per-element previews. Extracted content is automatically indexed for search.

**Image Segmentation** — Upload an image and describe what to segment in natural language. Granite Vision generates a coarse mask, refined by SAM for pixel-accurate results.

**Document Parsing** — Upload a document image or PDF to generate structured doctags output. View raw doctags and converted Markdown side-by-side, with per-page results for multi-page PDFs.

**Multipage QA** — Upload a PDF or up to 8 images and ask questions about the content. Images are resized to 768px max dimension for GPU memory efficiency. Answers are displayed alongside page thumbnails.

**Document Search** — Search across previously extracted document content using natural language questions. Uses Granite embedding model for semantic search and Granite Vision for RAG-powered answer generation. Sources are displayed alongside answers.

## Project Structure

```
streamlit_app.py         # navigation hub (st.navigation routing)
streamlit_home.py        # landing page with navigation cards
ui_helpers.py            # shared UI functions (preview, help, metrics, examples, sidebar)
pipeline/
  __init__.py            # public API re-exports
  models.py              # shared model loading, vision model factories, generate helper
  utils.py               # temp_upload and timed context managers
  config.py              # converter factory, convert wrapper
  output.py              # unified element builder, description and table extraction
  segmentation.py        # segmentation pipeline, SAM refinement, SAM model loader
  doctags.py             # doctags generation, parsing, PDF rendering
  qa.py                  # image resizing, multipage QA inference
  search.py              # embedding model, ChromaDB indexing, query, RAG generation
pages/
  extraction.py          # PDF extraction page
  segmentation.py        # image segmentation page
  doctags.py             # document parsing page
  qa.py                  # multipage QA page
  search.py              # document search page
examples/
  sample.jpg             # sample image for demo mode
  sample.pdf             # sample PDF for demo mode
tests/
  test_config.py         # converter factory and pipeline option tests
  test_output.py         # element builder, description, and table content tests
  test_models.py         # shared model loading and generate_response tests
  test_utils.py          # temp_upload and timed context manager tests
  test_segmentation.py   # SAM model, segment pipeline, and helper unit tests
  test_doctags.py        # doctags rendering, page count, parsing, and inference tests
  test_qa.py             # QA resizing, prompt structure, and validation tests
  test_search.py         # embedding, indexing, query, RAG, and collection tests
  test_ui_helpers.py     # UI helper functions and example file loading tests
```
