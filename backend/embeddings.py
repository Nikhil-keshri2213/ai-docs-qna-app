from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """Return a HuggingFace sentence-transformers embedding model (runs fully locally)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )