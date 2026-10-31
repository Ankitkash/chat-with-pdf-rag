from ingestion import load_pdf, chunk_text, build_vector_store

text = load_pdf("sample.pdf")
chunks = chunk_text(text)
vector_store = build_vector_store(chunks)

query = "What is the main advantage of using qudits over qubits as described in the document?"
results = vector_store.similarity_search(query, k=3)

for i, doc in enumerate(results, 1):
    print(f"\n--- Chunk {i} ---")
    print(doc.page_content)