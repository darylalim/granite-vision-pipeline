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
    format_qa_export,
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
    # Reset history only when a new file is uploaded
    current_file_id = (
        f"{getattr(uploaded_file, 'name', '')}_{getattr(uploaded_file, 'size', 0)}"
    )
    if st.session_state.get("current_file_id") != current_file_id:
        st.session_state["qa_history"] = []
        st.session_state["source_pages"] = []
        st.session_state["current_file_id"] = current_file_id
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
            assert isinstance(slider_range, tuple)

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
        st.session_state["current_file_name"] = getattr(
            uploaded_file, "name", "document.pdf"
        )
        st.session_state["current_page_range"] = (selected[0], selected[-1])

        # Reset conversation history if selection changed
        prev_sel = st.session_state.get("prev_selected")
        if prev_sel is not None and prev_sel != selected:
            st.session_state["qa_history"] = []
            st.session_state["source_pages"] = []
        st.session_state["prev_selected"] = selected

question = st.text_area(
    "Question",
    placeholder="e.g., What is shown on these pages?",
    height=100,
)
st.caption("Ctrl+Enter to submit")

has_input = bool(uploaded_file) and len(selected) >= 2 and bool(question)

if st.button("Answer", type="primary", disabled=not has_input):
    assert uploaded_file is not None

    if not st.session_state.get("model_granite_vision"):
        spinner_msg = "Loading model and generating answer..."
    else:
        spinner_msg = "Generating answer..."

    with st.spinner(spinner_msg):
        processor, model = qa_model()
        st.session_state["model_granite_vision"] = True

        with temp_upload(uploaded_file) as tmp_path:
            page_images = render_pdf_pages(
                tmp_path, page_indices=[i - 1 for i in selected]
            )

        # Store source pages (pre-resize) for verification tab
        st.session_state["source_pages"] = page_images

        with timed() as t:
            answer = generate_qa_response(page_images, question, processor, model)

    if not answer:
        st.warning("Model produced no output.")
    else:
        # Append to conversation history
        history = st.session_state.get("qa_history", [])
        history.append(
            {
                "question": question,
                "answer": answer,
                "duration_s": t.duration_s,
            }
        )
        st.session_state["qa_history"] = history

# Display results if there is history
history = st.session_state.get("qa_history", [])
source_pages = st.session_state.get("source_pages", [])

if history:
    tab_answer, tab_source = st.tabs(["Answer", "Source Pages"])

    with tab_answer:
        for entry in history:
            st.markdown(f"**Q:** {entry['question']}")
            with st.container(border=True):
                st.markdown(entry["answer"])
            duration = entry.get("duration_s")
            if duration is not None:
                st.caption(f"Generated in {duration:.2f}s")
            st.divider()

        # Download button — use session state for file metadata so it
        # persists across reruns even if the uploader widget resets
        file_name = st.session_state.get("current_file_name", "document.pdf")
        page_range_export = st.session_state.get("current_page_range")
        if page_range_export:
            export_md = format_qa_export(file_name, page_range_export, history)
            st.download_button(
                "Download Q&A",
                data=export_md,
                file_name="qa_export.md",
                mime="text/markdown",
            )

    with tab_source:
        if source_pages:
            src_cols_per_row = min(4, len(source_pages))
            page_numbers = st.session_state.get(
                "current_page_range", (1, len(source_pages))
            )
            src_page_list = list(range(page_numbers[0], page_numbers[1] + 1))
            for row_start in range(0, len(source_pages), src_cols_per_row):
                row_pages = list(
                    enumerate(
                        source_pages[row_start : row_start + src_cols_per_row],
                        start=row_start,
                    )
                )
                cols = st.columns(src_cols_per_row)
                for col, (idx, img) in zip(cols, row_pages):
                    page_num = (
                        src_page_list[idx] if idx < len(src_page_list) else idx + 1
                    )
                    col.image(img, caption=f"Page {page_num}", use_container_width=True)
        else:
            st.info("Source pages will appear here after generating an answer.")
