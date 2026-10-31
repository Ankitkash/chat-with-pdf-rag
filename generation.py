from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a document.
Use ONLY the context provided below to answer.
If the answer is not in the context, say: "I don't have enough information in this document to answer that."
Cite the relevant part of the context in your answer (for example, [Source 1])."""


def generate_answer(question, docs):
    context = "\n\n".join(
        f"[Source {i}]\n{doc.page_content}" for i, doc in enumerate(docs, 1)
    )
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    return llm.invoke(messages).content