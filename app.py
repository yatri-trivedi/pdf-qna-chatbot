import os
import streamlit as st
import tempfile
from ingest import ingest_pdf
from rag import load_qa_chain, ask

st.set_page_config(page_title='PDF Q&A BOT', page_icon="📄")
st.title("📄 PDF Question Answering Bot")
st.caption("Upload a PDF and ask questions about its contents")

os.environ['GROQ_API_KEY'] = st.sidebar.text_input(
    "GROQ API Key", type='password'
)

uploaded_file = st.file_uploader("Upload PDF", type='pdf')

if uploaded_file and os.environ.get("GROQ_API_KEY"):
    with st.spinner("Ingesting PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        vectorstore = ingest_pdf(tmp_path)
        chain_tuple = load_qa_chain()
        st.session_state["chain"] = chain_tuple
    st.success("PDF indexed! Ask away.")

if "chain" in st.session_state:
    query = st.text_input("Your question:")
    if query:
        with st.spinner("Thinking..."):
            result = ask(st.session_state["chain"], query)
        st.markdown("### Answer")
        st.write(result["answer"])
        with st.expander("Source chunks"):
            for src in result["sources"]:
                st.markdown(f"**Page {src['page']}**: {src['text']}…")