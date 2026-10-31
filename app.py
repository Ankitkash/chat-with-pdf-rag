import streamlit as st
from ingestion import load_pdf, chunk_text, build_vector_store
from generation import generate_answer

st.title("Chat with your PDF")

# Configurable k
k = st.sidebar.slider("Number of chunks to retrieve (k)", 1, 10, 3)

# Chat history storage
if "history" not in st.session_state:
    st.session_state.history = []

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded and st.session_state.get("file_name") != uploaded.name:
    with st.spinner("Processing PDF..."):
        chunks = chunk_text(load_pdf(uploaded))
        st.session_state.vector_store = build_vector_store(chunks)
        st.session_state.file_name = uploaded.name
        st.session_state.history = []
    st.success(f"Ready: {len(chunks)} chunks created")

# Show previous Q&A with sources
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        with st.expander("Sources"):
            for i, text in enumerate(item["sources"], 1):
                st.markdown(f"**Source {i}**")
                st.write(text)

question = st.chat_input("Ask a question about the PDF")

if question:
    if "vector_store" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        docs = st.session_state.vector_store.similarity_search(question, k=k)
        with st.spinner("Thinking..."):
            answer = generate_answer(question, docs)
        st.session_state.history.append(
            {
                "question": question,
                "answer": answer,
                "sources": [d.page_content for d in docs],
            }
        )
        st.rerun()