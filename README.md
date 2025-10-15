**RAG-Enhanced LLM Chat (Offline Chatbot with Document Understanding)**

This project is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, Ollama, and ChromaDB.

It allows you to chat with your documents (TXT, PDF, DOCX) completely offline — combining local AI models with your uploaded files.

<img width="1855" height="884" alt="image" src="https://github.com/user-attachments/assets/20ca2d27-1ab1-478b-a696-dfe70ea866f7" />
<img width="1770" height="795" alt="image" src="https://github.com/user-attachments/assets/d743b697-61fb-4ad2-aae9-975a586f9839" />
<img width="1842" height="869" alt="image" src="https://github.com/user-attachments/assets/d4089a23-58b8-4dbd-a0be-54aef3ee6b14" />
<img width="1844" height="859" alt="image" src="https://github.com/user-attachments/assets/d866b174-21d3-4e12-9a63-171db299d3ce" />

**Features**

Retrieval-Augmented Generation (RAG): Answers questions using your documents.

Document Support: Upload and process .txt, .pdf, or .docx files.

Ollama Integration: Works with locally hosted LLMs (e.g., gpt-oss:20b-cloud, llama3, etc.).

Persistent Vector Store: Saves embeddings using ChromaDB.

Interactive Chat Interface: Beautiful UI powered by Streamlit.

Customizable Settings: Toggle RAG, set model name, and clear chat or database easily.

**Requirements**

Make sure you have Python 3.10+ installed.
Then install dependencies:

pip install streamlit ollama chromadb sentence-transformers pypdf python-docx


You also need Ollama installed and running locally.
Download Ollama

Once installed, run:

ollama serve


To pull a model (example):

ollama pull gpt-oss:20b-cloud

**How to Run**

Clone or place this script in a folder:

app.py


Start Ollama server:

ollama serve


Run the Streamlit app:

streamlit run app.py


Open the local URL (usually http://localhost:8501) in your browser.

**Usage Guide**

Upload documents from the sidebar (supports .txt, .pdf, .docx).

Click “Process Documents” to store them in ChromaDB.

Enable RAG to use your uploaded content for question answering.

Type your question in the chat — the bot will combine your docs + model knowledge to answer.

You can also:

Change model name (Ollama model)

Clear chat

Clear knowledge base (ChromaDB)

**Project Structure**

rag-chatbot/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Documentation (you are here)
├── .gitignore             # Git ignore rules
│
├── setup.bat              # Windows setup script
├── setup.sh               # Mac/Linux setup script
├── start.bat              # Windows start script
├── start.sh               # Mac/Linux start script
│
├── venv/                  # Virtual environment (excluded from git)
├── chroma_db/             # Vector database storage (excluded from git)
│   ├── chroma.sqlite3     # SQLite database
│   └── ...                # Embedding data
│
└── __pycache__/           # Python cache (excluded from git)

**Notes**

Make sure Ollama is running before chatting.

If you change models, ensure it’s pulled using ollama pull <model_name>.

To delete all stored document embeddings, click “🗑️ Clear Knowledge Base” in the sidebar.

**Example Models (Ollama)**

gpt-oss:20b-cloud

llama3

mistral

phi3

qwen2

You can check installed models:

ollama list

