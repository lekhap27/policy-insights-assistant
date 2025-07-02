import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- CONFIG ---
PDF_FOLDER = "multi_pdf_agent/data"
st.set_page_config(page_title="📘 UNDP Policy Assistant", layout="wide")

# --- BACKGROUND IMAGE ---
def add_bg_image():
    st.markdown(
        '''
        <style>
        .stApp {
            background-image: url("https://nucleus.iaea.org/sites/connect/CGULSpublic/PublishingImages/undp%20logo.png");
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-size: 100px;
            background-position: 20px 20px;
            opacity: 0.97;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

add_bg_image()


# --- HEADER ---
st.title("📘 UNDP Policy Assistant")
st.subheader("Ask questions across all uploaded UNDP policy, HR, and admin documents. Get answers with file and page references.")

# --- LOAD PDFs ---
@st.cache_resource
def load_pdfs_from_folder(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            reader = PdfReader(path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    metadata = {"source": filename, "page": page_num + 1}
                    all_docs.append(Document(page_content=text, metadata=metadata))
    return all_docs

docs = load_pdfs_from_folder(PDF_FOLDER)

# --- SMART PDF SEARCH FILTER ---
filenames = list(set([doc.metadata["source"] for doc in docs]))
search_query = st.text_input("🔎 Filter PDFs by filename:", "")
filtered_files = [f for f in filenames if search_query.lower() in f.lower()]
filtered_docs = [doc for doc in docs if doc.metadata["source"] in filtered_files]

# --- DOWNLOAD SECTION ---
with st.expander("📁 View & Download Uploaded PDFs"):
    for file in sorted(filtered_files):
        file_path = os.path.join(PDF_FOLDER, file)
        st.markdown(f"📄 **{file}**")
        with open(file_path, "rb") as f:
            st.download_button(f"⬇️ Download {file}", f, file_name=file)

# --- BUILD QA CHAIN WITH MEMORY ---
@st.cache_resource
def create_qa_chain(_docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(_docs, embeddings)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

qa_chain = create_qa_chain(filtered_docs) if filtered_docs else None

# --- Initialize Session State for Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- QUESTION INPUT ---
query = st.text_input("❓ Ask a question about your documents:")

if query and qa_chain:
    with st.spinner("🤖 Thinking..."):
        result = qa_chain({"question": query})
        answer = result["answer"]
        sources = result["source_documents"]

        # Save interaction to session
        st.session_state.chat_history.append({
            "question": query,
            "answer": answer,
            "sources": sources
        })

# --- DISPLAY CHAT HISTORY ---
if st.session_state.chat_history:
    st.subheader("🧠 Chat History")
    for idx, entry in enumerate(st.session_state.chat_history[::-1], 1):
        st.markdown(f"**Q{len(st.session_state.chat_history) - idx + 1}:** {entry['question']}")
        st.markdown(f"**A:** {entry['answer']}")

        st.markdown("**📌 Sources:**")
        for i, doc in enumerate(entry["sources"]):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            content = doc.page_content.strip().replace("\n", " ")
            st.markdown(f"- `{source}` — Page {page}: {content[:300]}...")
        st.markdown("---")

# --- CLEAR CHAT BUTTON ---
if st.button("🗑️ Clear Conversation"):
    st.session_state.chat_history = []
    if qa_chain:
        qa_chain.memory.clear()
    st.experimental_rerun()
