from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

# Embedding Model
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


def create_vectorstore(pdf_file):
    # Load the uploaded PDF
    docs = PyPDFLoader(pdf_file).load()
    # Chunk it
    chunks = splitter.split_documents(docs)
    # Create vectorstore IN MEMORY — no persist_directory
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
        # No persist_directory — lives only for this session
    )
    return vectorstore