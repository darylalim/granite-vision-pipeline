import streamlit as st

from pipeline import (
    create_granite_vision_model,
    generate_qa_response,
    get_pdf_page_count,
    render_pdf_pages,
    temp_upload,
    timed,
)
from ui_helpers import (
    clamp_page_range,
    format_qa_export,  # noqa: F401
    load_example,
    render_thumbnail_grid,
    show_upload_preview,
)

st.set_page_config(page_title="Granite Vision Pipeline")

qa_model = st.cache_resource(create_granite_vision_model)

EXAMPLE_PDF = "examples/sample.pdf"

st.title("Multipage QA")
st.write(
    "Ask questions about document pages using IBM Granite Vision. "
    "Upload a PDF, select 2-8 consecutive pages, then type your question."
)

col_upload, col_example = st.columns([3, 1], vertical_alignment="bottom")
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=False,
    )
with col_example:
    if st.button("Try with example"):
        st.session_state["use_example_qa"] = True
        st.rerun()

# Resolve files: user upload takes priority over example
if uploaded_file:
    st.session_state.pop("use_example_qa", None)
elif st.session_state.get("use_example_qa"):
    uploaded_file = load_example(EXAMPLE_PDF)
    st.caption("Using example file")

selected: list[int] = []

if uploaded_file:
    show_upload_preview(uploaded_file)

    with temp_upload(uploaded_file) as path:
        total_pages = get_pdf_page_count(path)
    uploaded_file.seek(0)

    st.caption(f"{total_pages} pages")

    if total_pages < 2:
        st.error("PDF must have at least 2 pages.")
    else:
        # Render thumbnails at low DPI, cache in session state
        file_size = getattr(uploaded_file, "size", 0)
        file_key = (
            f"thumbs_{getattr(uploaded_file, 'name', '')}_{total_pages}_{file_size}"
        )
        if file_key not in st.session_state:
            with temp_upload(uploaded_file) as thumb_path:
                # Batch render for large PDFs to avoid blocking
                if total_pages > 50:
                    all_thumbs: list = []
                    for batch_start in range(0, total_pages, 20):
                        batch_end = min(batch_start + 20, total_pages)
                        batch_indices = list(range(batch_start, batch_end))
                        all_thumbs.extend(
                            render_pdf_pages(
                                thumb_path, dpi=72, page_indices=batch_indices
                            )
                        )
                    st.session_state[file_key] = all_thumbs
                else:
                    st.session_state[file_key] = render_pdf_pages(thumb_path, dpi=72)
            uploaded_file.seek(0)
        thumbnails = st.session_state[file_key]

        # Dynamic columns: 4 for small PDFs, 6 for larger ones
        cols_per_row = 6 if total_pages > 12 else 4

        # Range slider for consecutive page selection
        slider_key = "page_range_slider"
        if total_pages == 2:
            slider_range = (1, 2)
        else:
            if slider_key not in st.session_state:
                st.session_state[slider_key] = (1, min(total_pages, 8))
            slider_range = st.select_slider(
                "Select page range",
                options=list(range(1, total_pages + 1)),
                key=slider_key,
            )

        # Clamp to max 8 pages
        clamped = clamp_page_range(slider_range[0], slider_range[1], max_span=8)
        if clamped != tuple(slider_range):
            st.warning(
                f"Maximum 8 pages — selection narrowed to pages {clamped[0]}-{clamped[1]}"
            )
            st.session_state[slider_key] = clamped
            st.rerun()

        # Display thumbnail grid
        render_thumbnail_grid(
            thumbnails, selected_range=clamped, cols_per_row=cols_per_row
        )
        num_selected = clamped[1] - clamped[0] + 1
        st.caption(f"Pages {clamped[0]}-{clamped[1]} selected ({num_selected} pages)")

        selected = list(range(clamped[0], clamped[1] + 1))

        # Reset conversation history if selection changed
        prev_sel = st.session_state.get("prev_selected")
        if prev_sel != selected:
            st.session_state["qa_history"] = []
            st.session_state["source_pages"] = []
            st.session_state["prev_selected"] = selected

question = st.text_input("Question", placeholder="e.g., What is shown on these pages?")

has_input = bool(uploaded_file) and bool(selected) and bool(question)

if st.button("Answer", type="primary", disabled=not has_input):
    assert uploaded_file is not None
    processor, model = qa_model()
    st.session_state["model_granite_vision"] = True

    with temp_upload(uploaded_file) as tmp_path:
        with st.spinner("Rendering selected pages..."):
            page_images = render_pdf_pages(
                tmp_path, page_indices=[i - 1 for i in selected]
            )

        with st.spinner("Generating answer..."):
            with timed() as t:
                answer = generate_qa_response(page_images, question, processor, model)

    if not answer:
        st.warning("Model produced no output.")
    else:
        st.markdown(answer)
        st.caption(f"Generated in {t.duration_s:.2f}s")

    st.caption("Answers are limited to ~1024 tokens and may be truncated.")
