import os
from langchain_community.vectorstores import FAISS
from backend.embeddings import get_embedding_model

INDEX_PATH = "data/faiss_index"


def create_vector_store(chunks):
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_PATH)
    return vector_store


def load_vector_store():
    if not os.path.exists(INDEX_PATH):
        return None
    embeddings = get_embedding_model()
    return FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )


def update_vector_store(chunks):
    embeddings = get_embedding_model()
    if os.path.exists(INDEX_PATH):
        vector_store = FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_PATH)
    return vector_store


def get_retriever(vector_store, k=3):
    return vector_store.as_retriever(search_kwargs={"k": k})