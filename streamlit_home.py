import streamlit as st

st.title("Granite Vision Pipeline")
st.write("Document AI powered by IBM Granite models.")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("PDF Extraction")
        st.write("Extract and describe pictures and tables from PDF documents.")
        st.page_link(
            "pages/extraction.py", label="Open", icon=":material/picture_as_pdf:"
        )

with col2:
    with st.container(border=True):
        st.subheader("Image Segmentation")
        st.write("Segment objects in images using natural language prompts.")
        st.page_link(
            "pages/segmentation.py", label="Open", icon=":material/content_cut:"
        )

with col3:
    with st.container(border=True):
        st.subheader("Document Parsing")
        st.write("Parse document images to structured text in doctags format.")
        st.page_link("pages/doctags.py", label="Open", icon=":material/code:")

col4, col5 = st.columns(2)

with col4:
    with st.container(border=True):
        st.subheader("Multipage QA")
        st.write("Ask questions about document pages using vision AI.")
        st.page_link("pages/qa.py", label="Open", icon=":material/question_answer:")

with col5:
    with st.container(border=True):
        st.subheader("Document Search")
        st.write("Search across extracted content with RAG-powered answers.")
        st.page_link("pages/search.py", label="Open", icon=":material/search:")
