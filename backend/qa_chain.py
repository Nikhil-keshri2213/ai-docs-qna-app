from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def create_qa_chain(vector_store):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    return qa_chain