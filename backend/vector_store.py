from langchain_community.vectorstores import FAISS
from embeddings import get_embedding_model


def create_vector_store(chunks):
    """Embed document chunks and store them in an in-memory FAISS index."""
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store