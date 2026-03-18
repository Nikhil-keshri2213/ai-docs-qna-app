import streamlit as st
import requests

st.title("AI Document Q&A")

uploaded_file = st.file_uploader("Upload Document", type=["pdf"])

if uploaded_file:

    files = {"file": uploaded_file}

    response = requests.post(
        "http://localhost:8000/upload",
        files=files
    )

    st.success("Document uploaded successfully")

question = st.text_input("Ask a question")

if st.button("Ask"):

    response = requests.post(
        "http://localhost:8000/ask",
        params={"question": question}
    )

    answer = response.json()["answer"]

    st.write("Answer:")
    st.write(answer)