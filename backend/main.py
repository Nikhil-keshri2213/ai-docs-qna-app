from fastapi import FastAPI, UploadFile, File
import shutil

from document_loader import load_and_split_document
from vector_store import create_vector_store
from qa_chain import create_qa_chain

app = FastAPI()

vector_store = None
qa_chain = None


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    file_location = f"data/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = load_and_split_document(file_location)

    global vector_store
    vector_store = create_vector_store(chunks)

    global qa_chain
    qa_chain = create_qa_chain(vector_store)

    return {"message": "Document processed successfully"}


@app.post("/ask")
async def ask_question(question: str):

    response = qa_chain.run(question)

    return {"answer": response}