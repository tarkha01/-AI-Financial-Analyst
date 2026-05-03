# 📊 FinSmart AI — RAG-based Financial Document Assistant

> **LLM-Powered Financial Intelligence System** | Ask questions about any company's annual report using Retrieval-Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green?logo=chainlink)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Llama--3%20%7C%20Groq-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 What is FinSmart AI?

**FinSmart AI** lets you upload any company's financial PDF (10-K Annual Report, Earnings Report, etc.) and instantly chat with it using AI.

**Example workflow:**
1. Upload Apple's 100-page 10-K report
2. Ask: *"What are the company's main risk factors in 2024?"*
3. FinSmart AI reads the document and returns a precise, cited answer in seconds

No more manually scrolling through hundreds of pages — the AI does it for you.

---

## 🧠 Core Concept: RAG (Retrieval-Augmented Generation)

RAG is the architecture that makes this possible. Instead of feeding the entire document to the LLM (which is impossible due to token limits), we:

```
PDF Upload  ──►  Text Splitting  ──►  Embeddings  ──►  ChromaDB (Vector Store)
                                                               │
Your Question ──►  Similarity Search  ◄────────────────────────
                        │
                        ▼
                  Top-K Chunks  ──►  Llama-3 (Groq API)  ──►  Answer
```

| Step | What happens |
|------|-------------|
| **1. PDF Parsing** | PyPDFLoader extracts all raw text from every page |
| **2. Text Splitting** | Document is cut into overlapping chunks (~500 chars each) |
| **3. Embeddings** | Each chunk is converted into a numerical vector using `all-MiniLM-L6-v2` |
| **4. Vector Store** | Vectors are stored in ChromaDB for fast similarity search |
| **5. Retrieval** | Your question is embedded and the most relevant chunks are found |
| **6. Generation** | The top-K chunks + your question are sent to Llama-3, which generates the answer |

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **UI / Dashboard** | [Streamlit](https://streamlit.io) | Interactive web interface |
| **LLM Framework** | [LangChain](https://python.langchain.com) | Orchestrates the RAG pipeline |
| **Large Language Model** | [Llama-3 via Groq API](https://groq.com) | Generates natural language answers |
| **Embeddings** | [HuggingFace all-MiniLM-L6-v2](https://huggingface.co) | Converts text to vectors (free) |
| **Vector Database** | [ChromaDB](https://www.trychroma.com) | Stores & retrieves text chunks by similarity |
| **PDF Loader** | PyPDFLoader (LangChain) | Parses financial PDF documents |
| **Config** | python-dotenv | Manages API keys securely |

---

## 📁 Project Structure

```
FinSmart-AI/
│
├── app.py               # Main Streamlit application (full RAG pipeline)
├── requirements.txt     # All Python dependencies
├── .env                 # API keys (NOT committed to Git)
├── .gitignore           # Excludes .env, __pycache__, chroma_db/
└── README.md            # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- A free [Groq API Key](https://console.groq.com) (takes 30 seconds to get)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/finsmart-ai.git
cd finsmart-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the root folder:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 💡 Usage

1. **Upload** a financial PDF (e.g. Apple 10-K, Tesla Annual Report) via the sidebar
2. **Configure** model, chunk size, and retrieval settings
3. Click **"Process Document"** — the RAG pipeline runs automatically
4. **Ask questions** in the chat interface:
   - *"What are the main risk factors?"*
   - *"Summarize the revenue and net income"*
   - *"What is the company's growth strategy?"*
   - *"What does the company say about competition?"*
5. View **source chunks** used to generate each answer (full transparency)

---

## 🔑 Key Features

- ✅ **Zero cost** — uses free HuggingFace embeddings + free Groq API tier
- ✅ **Any financial PDF** — works with 10-K, 10-Q, annual reports, earnings calls
- ✅ **Source transparency** — shows exact document chunks used for each answer
- ✅ **Configurable pipeline** — tune chunk size, overlap, top-K, temperature
- ✅ **Multiple LLMs** — switch between Llama-3 8B, Llama-3 70B, Mixtral
- ✅ **Suggested questions** — one-click starter prompts for financial analysis
- ✅ **Chat history** — full conversation context within the session

---

## 📊 Example Questions to Try

| Document | Question |
|----------|---------|
| Apple 10-K | *"What were Apple's total net sales in fiscal 2024?"* |
| Tesla Annual | *"What risks does Tesla identify related to EV competition?"* |
| Google 10-K | *"How does Alphabet generate revenue from Google Services?"* |
| Any report | *"What is the company's outlook for the next fiscal year?"* |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using LangChain, Groq, ChromaDB, and Streamlit*
