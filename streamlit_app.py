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
    load_example,
    render_thumbnail_grid,
    show_upload_preview,
)

st.set_page_config(page_title="PDF Question & Answer")

qa_model = st.cache_resource(create_granite_vision_model)

EXAMPLE_PDF = "examples/sample.pdf"

st.title("PDF Question & Answer")
st.write("Upload a PDF, then ask a question.")

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
    # Clear stale answer when file changes
    new_file_id = (
        f"{getattr(uploaded_file, 'name', '')}_{getattr(uploaded_file, 'size', 0)}"
    )
    if st.session_state.get("_upload_file_id") != new_file_id:
        st.session_state["_upload_file_id"] = new_file_id
        # Clean up old thumbnail cache
        old_thumb_key = st.session_state.get("_thumb_key")
        if old_thumb_key:
            st.session_state.pop(old_thumb_key, None)
        for key in (
            "last_answer",
            "last_duration_s",
            "last_source_pages",
            "last_page_numbers",
        ):
            st.session_state.pop(key, None)
elif st.session_state.get("use_example_qa"):
    uploaded_file = load_example(EXAMPLE_PDF)
    st.caption("Using example file")

selected: list[int] = []

if uploaded_file:
    show_upload_preview(uploaded_file)

    with temp_upload(uploaded_file) as path:
        total_pages = get_pdf_page_count(path)
    uploaded_file.seek(0)

    # Auto-select first N pages (N = min(total_pages, 8))
    auto_end = min(total_pages, 8)
    selected = list(range(1, auto_end + 1))

    if total_pages > 8:
        # Show page override expander for large PDFs
        # Use persisted slider value if available, otherwise auto-selected default
        persisted = st.session_state.get("page_range_slider")
        if isinstance(persisted, tuple) and len(persisted) == 2:
            display_start, display_end = persisted
        else:
            display_start, display_end = selected[0], selected[-1]
        st.caption(
            f"{total_pages} pages — Pages {display_start}-{display_end} selected"
        )

        with st.expander("Change pages"):
            # Cache thumbnails in session state
            file_size = getattr(uploaded_file, "size", 0)
            file_key = (
                f"thumbs_{getattr(uploaded_file, 'name', '')}_{total_pages}_{file_size}"
            )
            st.session_state["_thumb_key"] = file_key
            if file_key not in st.session_state:
                with temp_upload(uploaded_file) as thumb_path:
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
                        st.session_state[file_key] = render_pdf_pages(
                            thumb_path, dpi=72
                        )
                uploaded_file.seek(0)
            thumbnails = st.session_state[file_key]

            file_id = f"{getattr(uploaded_file, 'name', '')}_{total_pages}_{file_size}"
            if st.session_state.get("_slider_file_id") != file_id:
                st.session_state.pop("page_range_slider", None)
                st.session_state["_slider_file_id"] = file_id
            slider_key = "page_range_slider"
            slider_range = st.select_slider(
                "Select page range",
                options=list(range(1, total_pages + 1)),
                value=(1, min(total_pages, 8)),
                key=slider_key,
            )
            if not isinstance(slider_range, tuple):
                slider_range = (slider_range, slider_range)

            clamped = clamp_page_range(slider_range[0], slider_range[1], max_span=8)
            if clamped != tuple(slider_range):
                st.warning(
                    f"Maximum 8 pages — selection narrowed to pages {clamped[0]}-{clamped[1]}"
                )
                st.session_state[slider_key] = clamped
                st.rerun()

            cols_per_row = 6 if total_pages > 12 else 4
            render_thumbnail_grid(
                thumbnails, selected_range=clamped, cols_per_row=cols_per_row
            )

            selected = list(range(clamped[0], clamped[1] + 1))
            st.caption(
                f"Pages {clamped[0]}-{clamped[1]} selected ({len(selected)} pages)"
            )
    else:
        st.caption(
            f"{total_pages} page{'s' if total_pages != 1 else ''} — All pages selected"
        )

question = st.text_area(
    "Question",
    placeholder="e.g., What is shown on these pages?",
    height=100,
)

if st.button("Answer", type="primary"):
    if not uploaded_file:
        st.warning("Upload a PDF first.")
        st.stop()
    if not selected:
        st.warning("No pages selected.")
        st.stop()
    if not question:
        st.warning("Enter a question first.")
        st.stop()

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

        with timed() as t:
            answer = generate_qa_response(page_images, question, processor, model)

    if not answer:
        st.warning("Model produced no output.")
    else:
        st.session_state["last_answer"] = answer
        st.session_state["last_duration_s"] = t.duration_s
        st.session_state["last_source_pages"] = page_images
        st.session_state["last_page_numbers"] = selected

# Display result
answer = st.session_state.get("last_answer")
if answer:
    with st.container(border=True):
        st.markdown(answer)
    duration = st.session_state.get("last_duration_s")
    if duration is not None:
        st.caption(f"Generated in {duration:.2f}s")

    source_pages = st.session_state.get("last_source_pages", [])
    page_numbers = st.session_state.get("last_page_numbers", [])
    if source_pages:
        st.divider()
        cols_per_row = min(4, len(source_pages))
        for row_start in range(0, len(source_pages), cols_per_row):
            row_pages = list(
                enumerate(
                    source_pages[row_start : row_start + cols_per_row],
                    start=row_start,
                )
            )
            cols = st.columns(cols_per_row)
            for col, (idx, img) in zip(cols, row_pages):
                page_num = page_numbers[idx] if idx < len(page_numbers) else idx + 1
                col.image(img, caption=f"Page {page_num}", width="stretch")
