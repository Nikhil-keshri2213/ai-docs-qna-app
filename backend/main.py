"""
main.py — optional FastAPI backend
Run with:  uvicorn main:app --reload
"""

import shutil
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv

from document_loader import load_and_split_document
from vector_store import create_vector_store
from qa_chain import create_qa_chain

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("pdf_qa_api")

# ---------------------- App Setup ----------------------
load_dotenv()

app = FastAPI(title="PDF Q&A API")

UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory state (single-user; for multi-user use session-based storage)
vector_store = None
qa_chain = None


# ---------------------- Routes ----------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store, qa_chain

    logger.info(f"Received upload request: {file.filename}")

    file_path = UPLOAD_DIR / file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved at: {file_path}")

        chunks = load_and_split_document(str(file_path))
        logger.info(f"Document split into {len(chunks)} chunks")

        vector_store = create_vector_store(chunks)
        logger.info("Vector store created")

        qa_chain = create_qa_chain(vector_store)
        logger.info("QA chain initialized")

        return {
            "message": f"Document '{file.filename}' processed — {len(chunks)} chunks indexed."
        }

    except Exception as e:
        logger.exception("Error during document upload and processing")
        return {"error": str(e)}


@app.post("/ask")
async def ask_question(question: str):
    logger.info(f"Received question: {question}")

    if qa_chain is None:
        logger.warning("QA chain not initialized — no document uploaded")
        return {"answer": "No document uploaded yet."}

    try:
        result = qa_chain.invoke({"query": question})

        logger.info("Answer generated successfully")

        source_pages = [
            doc.metadata.get("page", "?")
            for doc in result.get("source_documents", [])
        ]

        logger.debug(f"Source pages: {source_pages}")

        return {
            "answer": result["result"],
            "source_pages": source_pages,
        }

    except Exception as e:
        logger.exception("Error while processing question")
        return {"error": str(e)}