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
