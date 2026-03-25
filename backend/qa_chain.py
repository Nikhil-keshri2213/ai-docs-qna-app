import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms import HuggingFaceEndpoint

log = logging.getLogger("docmind")

# Serverless HF model (no local download)
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

PROMPT_TEMPLATE = """You are a helpful assistant.

Answer ONLY from the provided context.
If the answer is not in the context, say:
"I don't know based on the document."

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(vector_store):
    """
    RAG pipeline using Hugging Face Inference API (no local model download)
    """
    log.info(f"Using Hugging Face hosted model: {LLM_MODEL}")

    # 🔥 Remote model (no download)
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        temperature=0.1,
        max_new_tokens=256,
        huggingfacehub_api_token=None,  # auto-picks from env
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    log.info("QA chain ready (remote inference)")

    return chain