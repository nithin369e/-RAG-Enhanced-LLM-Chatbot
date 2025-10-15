# Install required libraries first:
# pip install streamlit ollama chromadb sentence-transformers pypdf python-docx

import streamlit as st
from datetime import datetime
import ollama
import chromadb
from chromadb.utils import embedding_functions
import os
import tempfile
from pathlib import Path

# Document processing imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="RAG-Enhanced LLM Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e1e;
    }
    .stChatMessage {
        background-color: #2b2b2b;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .stTextInput > div > div > input {
        background-color: #2b2b2b;
        color: white;
    }
    .context-box {
        background-color: #2b2b2b;
        border-left: 3px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize ChromaDB
@st.cache_resource
def init_chroma():
    """Initialize ChromaDB client and collection"""
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use sentence transformers for embeddings
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    return client, collection

# Document processing functions
def extract_text_from_txt(file):
    """Extract text from TXT file"""
    return file.read().decode('utf-8')

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        return "PyPDF2 not installed. Install with: pip install pypdf2"
    
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    if not DOCX_AVAILABLE:
        return "python-docx not installed. Install with: pip install python-docx"
    
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def process_document(file, collection):
    """Process uploaded document and add to vector store"""
    file_extension = file.name.split('.')[-1].lower()
    
    # Extract text based on file type
    if file_extension == 'txt':
        text = extract_text_from_txt(file)
    elif file_extension == 'pdf':
        text = extract_text_from_pdf(file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(file)
    else:
        return False, "Unsupported file type"
    
    # Chunk the text
    chunks = chunk_text(text)
    
    if not chunks:
        return False, "No text could be extracted from the document"
    
    # Add chunks to vector store
    ids = [f"{file.name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file.name, "chunk_id": i} for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    
    return True, f"Successfully added {len(chunks)} chunks from {file.name}"

def retrieve_context(query, collection, n_results=3):
    """Retrieve relevant context from vector store"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if results['documents'] and results['documents'][0]:
            contexts = results['documents'][0]
            sources = [meta['source'] for meta in results['metadatas'][0]]
            return contexts, sources
        return [], []
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return [], []

def create_rag_prompt(query, contexts):
    """Create a prompt with retrieved context"""
    context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, say so and provide a general answer based on your knowledge.

Context Information:
{context_text}

User Question: {query}

Answer:"""
    
    return prompt

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-oss:20b-cloud"

if "use_rag" not in st.session_state:
    st.session_state.use_rag = True

if "n_results" not in st.session_state:
    st.session_state.n_results = 3

# Initialize ChromaDB
client, collection = init_chroma()

# Sidebar for settings and document upload
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_input = st.text_input(
        "Model Name",
        value=st.session_state.model_name,
        help="Enter the Ollama model name"
    )
    if model_input != st.session_state.model_name:
        st.session_state.model_name = model_input
        st.rerun()
    
    # RAG toggle
    st.session_state.use_rag = st.checkbox(
        "Enable RAG",
        value=st.session_state.use_rag,
        help="Use Retrieval Augmented Generation"
    )
    
    # Number of context chunks
    st.session_state.n_results = st.slider(
        "Context Chunks",
        min_value=1,
        max_value=5,
        value=st.session_state.n_results,
        help="Number of relevant chunks to retrieve"
    )
    
    st.divider()
    
    # Document upload section
    st.header("📁 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['txt', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="Upload documents to add to knowledge base"
    )
    
    if uploaded_files:
        if st.button("Process Documents", use_container_width=True):
            progress_bar = st.progress(0)
            success_count = 0
            
            for idx, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    success, message = process_document(file, collection)
                    if success:
                        st.success(message)
                        success_count += 1
                    else:
                        st.error(message)
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            st.info(f"Processed {success_count}/{len(uploaded_files)} documents successfully")
    
    st.divider()
    
    # Collection info
    st.subheader("📊 Knowledge Base")
    try:
        count = collection.count()
        st.metric("Total Chunks", count)
    except:
        st.metric("Total Chunks", 0)
    
    # Clear collection
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        try:
            client.delete_collection("documents")
            st.success("Knowledge base cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Info section
    st.markdown("""
    ### 📝 Instructions
    1. Upload documents (TXT, PDF, DOCX)
    2. Click "Process Documents"
    3. Enable RAG to use your documents
    4. Ask questions about your content
    
    ### 🔧 Requirements
    - Ollama must be running
    - Model must be installed
    
    Run: `ollama serve`
    """)

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.title("🤖 RAG-Enhanced LLM Chatbox")

with col2:
    status_color = "🟢" if st.session_state.use_rag else "🟡"
    st.caption(f"{status_color} RAG: {'ON' if st.session_state.use_rag else 'OFF'}")

st.caption(f"Model: {st.session_state.model_name} | Status: Online ✓")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show retrieved context if available
        if message["role"] == "assistant" and "contexts" in message and message["contexts"]:
            with st.expander("📚 Retrieved Context"):
                for i, (ctx, source) in enumerate(zip(message["contexts"], message["sources"])):
                    st.markdown(f"**Source: {source}**")
                    st.markdown(f'<div class="context-box">{ctx[:300]}...</div>', unsafe_allow_html=True)
        
        st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)
    
    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        context_placeholder = st.empty()
        full_response = ""
        contexts = []
        sources = []
        
        try:
            # Retrieve context if RAG is enabled
            if st.session_state.use_rag:
                with st.spinner("Searching knowledge base..."):
                    contexts, sources = retrieve_context(
                        prompt, 
                        collection, 
                        n_results=st.session_state.n_results
                    )
                
                if contexts:
                    # Create RAG prompt
                    rag_prompt = create_rag_prompt(prompt, contexts)
                    query_content = rag_prompt
                else:
                    query_content = prompt
                    st.info("No relevant context found in knowledge base. Using general knowledge.")
            else:
                query_content = prompt
            
            # Create conversation history
            conversation_history = [
                {"role": "user", "content": query_content}
            ]
            
            # Get response from LLM
            with st.spinner("Thinking..."):
                response = ollama.chat(
                    model=st.session_state.model_name,
                    messages=conversation_history
                )
                
                full_response = response['message']['content']
                message_placeholder.markdown(full_response)
            
            # Show retrieved context
            if contexts:
                with context_placeholder.expander("📚 Retrieved Context", expanded=False):
                    for i, (ctx, source) in enumerate(zip(contexts, sources)):
                        st.markdown(f"**Source: {source}**")
                        st.markdown(f'<div class="context-box">{ctx[:300]}...</div>', unsafe_allow_html=True)
            
            # Add timestamp
            response_timestamp = datetime.now().strftime("%H:%M:%S")
            st.caption(response_timestamp)
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": response_timestamp,
                "contexts": contexts,
                "sources": sources
            })
            
        except Exception as e:
            error_msg = f"""
            ⚠️ **Error occurred:**
            
            {str(e)}
            
            **Possible solutions:**
            - Make sure Ollama is running: `ollama serve`
            - Check if model exists: `ollama list`
            - Pull the model: `ollama pull {st.session_state.model_name}`
            """
            message_placeholder.markdown(error_msg)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

# Welcome message for first time users
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        welcome_msg = """👋 Welcome! I'm your RAG-enhanced AI assistant running completely offline. 

I can answer questions based on:
- Documents you upload (TXT, PDF, DOCX)
- My general knowledge

**Get started:**
1. Upload documents in the sidebar
2. Click "Process Documents"
3. Enable RAG and start chatting!"""
        st.markdown(welcome_msg)
        st.caption("Ready to chat")