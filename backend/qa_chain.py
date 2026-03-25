import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ✅ REPLACE THIS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


PROMPT_TEMPLATE = """You are a helpful assistant.

Answer ONLY from the provided context.
If not found, say:
"I don't know based on the document."

Context:
{context}

Question: {question}

Answer:
"""


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(vector_store):

    # ✅ LOCAL MODEL (NO API, NO TOKEN)
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256,
        temperature=0.1
    )

    llm = HuggingFacePipeline(pipeline=pipe)

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

    return chain