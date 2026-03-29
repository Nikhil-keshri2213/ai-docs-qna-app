from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    return text.replace("\n", " ").strip()


def load_and_split_document(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        return []

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        if "page" not in doc.metadata:
            doc.metadata["page"] = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    return chunks