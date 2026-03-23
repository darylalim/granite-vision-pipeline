import streamlit as st

from pipeline import (
    clear_collection,
    create_embedding_model,
    create_granite_vision_model,
    generate_answer,
    get_collection,
    query_index,
)

embedding_model = st.cache_resource(create_embedding_model)
collection = st.cache_resource(get_collection)
qa_model = st.cache_resource(create_granite_vision_model)

st.title("Document Search")
st.write(
    "Search across extracted document content using natural language questions. "
    "Documents are automatically indexed when processed on the PDF Extraction page."
)

coll = collection()


@st.cache_data(ttl=5)
def _get_doc_count() -> int:
    return coll.count()


doc_count = _get_doc_count()
st.metric("Indexed documents", doc_count)

if doc_count == 0:
    st.info("No documents indexed yet. Extract a PDF first.")

question = st.text_input(
    "Question", placeholder="e.g., What does the revenue chart show?"
)

if st.button("Search", type="primary", disabled=not question or doc_count == 0):
    model = embedding_model()

    with st.spinner("Searching..."):
        results = query_index(question, model, coll)

    if not results:
        st.warning("No relevant results found for your question.")
    else:
        processor, gen_model = qa_model()

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
