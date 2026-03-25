import shutil
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv

from backend.document_loader import load_and_split_document
from backend.vector_store import create_vector_store
from backend.qa_chain import create_qa_chain

# ---------------------- Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("pdf_qa_api")

# ---------------------- App ----------------------
load_dotenv()
app = FastAPI(title="PDF Q&A API")

UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory (single user)
vector_store = None
qa_chain = None
retriever = None   # ✅ NEW


# ---------------------- Upload ----------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store, qa_chain, retriever

    logger.info(f"Upload: {file.filename}")

    file_path = UPLOAD_DIR / file.filename

    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Saved: {file_path}")

        # Process
        chunks = load_and_split_document(str(file_path))
        logger.info(f"Chunks: {len(chunks)}")

        vector_store = create_vector_store(chunks)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        qa_chain = create_qa_chain(vector_store)

        logger.info("Pipeline ready")

        return {
            "message": f"{file.filename} processed",
            "chunks": len(chunks)
        }

    except Exception as e:
        logger.exception("Upload failed")
        return {"error": str(e)}


# ---------------------- Ask ----------------------
@app.post("/ask")
async def ask_question(question: str):
    global qa_chain, retriever

    logger.info(f"Question: {question}")

    if qa_chain is None:
        return {"answer": "Upload document first."}

    try:
        # ✅ FIX: chain takes string
        answer = qa_chain.invoke(question)

        # ✅ FIX: fetch sources separately
        docs = retriever.invoke(question)

        source_pages = [
            doc.metadata.get("page", "?") + 1
            for doc in docs
        ]

        return {
            "answer": answer,
            "source_pages": source_pages
        }

    except Exception as e:
        logger.exception("QA failed")
        return {"error": str(e)}