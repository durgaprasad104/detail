import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- CPU-Optimized Configuration ---
DATA_FOLDER = "documents"
EMBED_MODEL = "sentence-transformers/all-miniLM-L6-v2"  # Smaller embedding model
LLM_MODEL = "google/flan-t5-large"  # Medium-sized CPU-friendly model; consider using a smaller model if memory is still an issue
CHUNK_SIZE = 256  # Reduced for CPU efficiency
MAX_TOKENS = 400

# --- Document Processing ---
def load_documents():
    docs = []
    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())
    return docs

text_splitter = SentenceTransformersTokenTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=50
)
documents = load_documents()
docs_chunks = text_splitter.split_documents(documents)

# --- Vector Store (Simplified for CPU) ---
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_store = FAISS.from_documents(docs_chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- CPU-Optimized LLM Setup ---
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
# Use low_cpu_mem_usage to reduce memory requirements.
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL, low_cpu_mem_usage=True)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=MAX_TOKENS,
    temperature=0.3,
    device=-1  # Force CPU usage
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- Streamlit Interface ---
st.title("🔍 CPU-Friendly Q&A Bot")

def format_answer(context, query):
    prompt = f"""Answer the question using only the context below. 
If unsure, say "I don't know".

Context: {context}

Question: {query}

Answer:"""
    return llm(prompt)

if 'query' not in st.session_state:
    st.session_state.query = ""

query = st.text_input("Ask a question:", value=st.session_state.query)

if st.button("Search") or st.session_state.query:
    # Retrieve relevant documents (up to 3)
    context_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in context_docs])[:2000]

    answer = format_answer(context, query)

    st.subheader("Answer:")
    st.write(answer)

    # Removed: Source contexts and next question suggestions to simplify the interface.
