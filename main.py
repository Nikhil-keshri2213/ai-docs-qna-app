import uuid
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.document_loader import load_and_split_document
from backend.vector_store import update_vector_store, load_vector_store
from backend.qa_chain import create_qa_chain
from core.config import settings

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("docmind_api")

app = FastAPI(title=settings.PROJECT_NAME)

UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

# Shared mutable state (module-level, safe for single-worker Uvicorn)
_state = {
    "vector_store": None,
    "qa_chain": None,
}

# ── Schemas ───────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    source_pages: List[int]

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load existing FAISS index on startup (if available)."""
    vector_store = load_vector_store()
    if vector_store:
        _state["vector_store"] = vector_store
        _state["qa_chain"] = create_qa_chain()
        logger.info("✅ Existing FAISS index loaded — ready to answer questions")
    else:
        logger.info("ℹ️  No existing index found — upload a PDF to get started")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}!",
        "status": "ready" if _state["vector_store"] else "no_document",
        "docs": "/docs"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved upload → {file_path}")

        # 1. Load & split
        chunks = load_and_split_document(str(file_path))
        if not chunks:
            raise HTTPException(status_code=400, detail="PDF has no readable text.")

        # 2. Update vector store (merges with existing index)
        _state["vector_store"] = update_vector_store(chunks)
        logger.info(f"Indexed {len(chunks)} chunks")

        # 3. (Re-)create QA chain — model is cached so this is fast
        _state["qa_chain"] = create_qa_chain()

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": len(chunks),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(payload: QuestionRequest):
    if not _state["vector_store"] or not _state["qa_chain"]:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a PDF first."
        )

    try:
        question = payload.question.strip()

        # 1. Retrieve top-k chunks WITH relevance scores (L2 distance — lower = more similar)
        scored_docs = _state["vector_store"].similarity_search_with_score(question, k=3)

        if not scored_docs:
            return AnswerResponse(
                answer="Asked Question is out of my scope",
                source_pages=[]
            )

        # 2. Relevance guard — FAISS L2 distance; anything above 1.0 is a poor match
        RELEVANCE_THRESHOLD = 1.0
        relevant_docs = [doc for doc, score in scored_docs if score <= RELEVANCE_THRESHOLD]

        if not relevant_docs:
            logger.info(f"All chunks scored above threshold ({[round(s,3) for _,s in scored_docs]}) — out of scope")
            return AnswerResponse(
                answer="Asked Question is out of my scope",
                source_pages=[]
            )

        docs = relevant_docs

        # 3. Build context string
        context = "\n\n".join(doc.page_content for doc in docs)

        # 4. Generate answer
        raw_answer: str = _state["qa_chain"].invoke({
            "context": context,
            "question": question
        })

        # 5. Clean up answer
        answer = raw_answer.strip()

        # Remove prompt-echoing artefacts from some models
        for prefix in ("answer:", "helpful answer:", "answer :"):
            if answer.lower().startswith(prefix):
                answer = answer[len(prefix):].strip()

        # Catch the "unanswerable" signal word we taught the model to emit
        OUT_OF_SCOPE_SIGNALS = {"unanswerable", "i don't know", "i do not know", "not found", ""}
        if answer.lower() in OUT_OF_SCOPE_SIGNALS:
            answer = "Asked Question is out of my scope"

        # 5. Collect unique source page numbers (1-indexed)
        source_pages = sorted({
            int(doc.metadata.get("page", 0)) + 1
            for doc in docs
            if doc.metadata.get("page") is not None
        })

        return AnswerResponse(answer=answer, source_pages=source_pages)

    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="AI is currently unavailable.")


@app.delete("/reset")
async def reset_index():
    """Clear the in-memory state (useful for testing without restarting)."""
    _state["vector_store"] = None
    _state["qa_chain"] = None
    logger.info("State reset")
    return {"status": "reset"}