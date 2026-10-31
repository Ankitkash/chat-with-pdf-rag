from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return splitter.split_text(text)

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_texts(chunks, embedding=embeddings)


if __name__ == "__main__":
    text = load_pdf("sample.pdf")
    chunks = chunk_text(text)
    print("Number of chunks:", len(chunks))
    vector_store = build_vector_store(chunks)
    print("Vector store created")