from langchain_community.vectorstores import FAISS
from backend.embeddings import get_embedding_model   # ✅ fixed import


def create_vector_store(chunks):
    embeddings = get_embedding_model()
    return FAISS.from_documents(chunks, embeddings)