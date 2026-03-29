"""
QA Chain — 100% FREE, runs locally via HuggingFace transformers.

Model: google/flan-t5-base  (~900MB, seq2seq, great for Q&A)
- No API key required
- Downloads once and caches at ~/.cache/huggingface/
- CPU-compatible, ~3-8s per query on CPU

To use a larger/better model (still free, just slower):
  Change MODEL_ID to "google/flan-t5-large" or "google/flan-t5-xl"
"""

import torch
from functools import lru_cache

from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "google/flan-t5-base"

PROMPT_TEMPLATE = """You are a document assistant. Read the context and answer the question.
Only use information from the context. Do not guess or add extra information.
If the context does not contain the answer, output only the word: unanswerable

Context: {context}

Question: {question}

Answer:"""


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load model once and cache it for the lifetime of the process."""
    device = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,          # deterministic
        temperature=1.0,          # ignored when do_sample=False, set to avoid warning
        device=device,
    )
    return pipe


def create_qa_chain():
    """
    Returns an LCEL chain: PromptTemplate | HuggingFacePipeline | StrOutputParser
    Call with: chain.invoke({"context": "...", "question": "..."})
    """
    pipe = _load_pipeline()
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = prompt | llm | StrOutputParser()
    return chain