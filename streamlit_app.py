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
    load_example,
    show_help,
    show_metrics_bar,
    show_sidebar_status,
    show_upload_preview,
)

st.set_page_config(page_title="Granite Vision Pipeline")

qa_model = st.cache_resource(create_granite_vision_model)

EXAMPLE_PDF = "examples/sample.pdf"

st.title("Multipage QA")
st.write(
    "Ask questions about document pages using IBM Granite Vision. "
    "Upload a PDF, then type your question."
)

show_help(
    supported_formats="PDF",
    description=(
        "Upload a PDF and select which pages to include (up to 8). "
        "Type a question and the model will analyze all selected pages "
        "together to generate an answer. Images are resized to "
        "768px max dimension to fit within memory limits."
    ),
    model_info="[granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b)",
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

    default_pages = list(range(1, min(9, total_pages + 1)))
    selected = st.multiselect(
        "Select pages (up to 8)",
        options=list(range(1, total_pages + 1)),
        default=default_pages,
        max_selections=8,
    )

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
        col_thumbs, col_answer = st.columns([1, 2])
        with col_thumbs:
            for i, img in enumerate(page_images):
                st.image(
                    img,
                    caption=f"Page {selected[i]}",
                    width="stretch",
                )
        with col_answer:
            st.markdown(answer)

        show_metrics_bar({"Duration (s)": f"{t.duration_s:.2f}"})

    st.caption("Answers are limited to ~1024 tokens and may be truncated.")

show_sidebar_status(
    models={"Granite Vision": st.session_state.get("model_granite_vision", False)},
)
