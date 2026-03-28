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

uploaded_file = st.file_uploader(
    "Upload a PDF to get started",
    type=["pdf"],
    accept_multiple_files=False,
)
if st.button("Try with example", type="tertiary"):
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
        # Use persisted slider value for display, fall back to auto-selected
        persisted = st.session_state.get("page_range_slider")
        if isinstance(persisted, tuple) and len(persisted) == 2:
            display_start, display_end = persisted
        else:
            display_start, display_end = selected[0], selected[-1]

        picker_key = "show_page_picker"
        if st.button(
            f"Pages {display_start}–{display_end} of {total_pages} ✎",
            type="tertiary",
            key="page_range_toggle",
        ):
            st.session_state[picker_key] = not st.session_state.get(picker_key, False)
            st.rerun()

        if st.session_state.get(picker_key, False):
            with st.container(border=True):
                st.caption("Select up to 8 pages")
                file_size = getattr(uploaded_file, "size", 0)
                file_id = (
                    f"{getattr(uploaded_file, 'name', '')}_{total_pages}_{file_size}"
                )
                if st.session_state.get("_slider_file_id") != file_id:
                    st.session_state.pop("page_range_slider", None)
                    st.session_state["_slider_file_id"] = file_id

                slider_range = st.select_slider(
                    "Select page range",
                    options=list(range(1, total_pages + 1)),
                    value=(1, min(total_pages, 8)),
                    key="page_range_slider",
                    label_visibility="collapsed",
                )
                if not isinstance(slider_range, tuple):
                    slider_range = (slider_range, slider_range)

                clamped = clamp_page_range(slider_range[0], slider_range[1], max_span=8)
                if clamped != tuple(slider_range):
                    st.warning(
                        f"Maximum 8 pages — selection narrowed to pages {clamped[0]}–{clamped[1]}"
                    )
                    st.session_state["page_range_slider"] = clamped
                    st.rerun()

                selected = list(range(clamped[0], clamped[1] + 1))
                st.caption(
                    f"Pages {clamped[0]}–{clamped[1]} selected ({len(selected)} pages)"
                )
    else:
        st.caption(
            f"{total_pages} page{'s' if total_pages != 1 else ''} — All pages selected"
        )

if uploaded_file:
    col_q, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    with col_q:
        question = st.text_input(
            "Question",
            placeholder="Ask a question about this PDF...",
            label_visibility="collapsed",
        )
    with col_btn:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    if ask_clicked:
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
        source_pages = st.session_state.get("last_source_pages", [])
        page_numbers = st.session_state.get("last_page_numbers", [])

        # Footer row: metadata left, toggle right
        col_meta, col_toggle = st.columns([3, 1])
        with col_meta:
            meta_parts: list[str] = []
            if duration is not None:
                meta_parts.append(f"{duration:.1f}s")
            if page_numbers:
                meta_parts.append(f"Pages {page_numbers[0]}–{page_numbers[-1]}")
            if meta_parts:
                st.caption(" · ".join(meta_parts))

        with col_toggle:
            if source_pages:
                toggle_key = "show_source_pages"
                if st.button(
                    "Hide source pages"
                    if st.session_state.get(toggle_key)
                    else "Show source pages",
                    type="tertiary",
                    key="source_toggle",
                ):
                    st.session_state[toggle_key] = not st.session_state.get(
                        toggle_key, False
                    )
                    st.rerun()

        if source_pages and st.session_state.get("show_source_pages", False):
            render_thumbnail_grid(
                source_pages,
                selected_range=(1, len(source_pages)),
                cols_per_row=min(6, len(source_pages)),
            )
