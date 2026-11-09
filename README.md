# RAG System with Gemini Embeddings

A production-ready Retrieval-Augmented Generation (RAG) system powered by Google's Gemini API, featuring document processing, link enrichment, intelligent reranking, and web grounding capabilities.

## 🚀 Features

### Core Capabilities
- **Document Processing**: Upload and process text documents with automatic chunking
- **Smart Link Processing**: Automatically extracts and indexes content from URLs and PDFs found in documents
- **Gemini Embeddings**: Uses `gemini-embedding-001` (3072 dimensions) for high-quality vector representations
- **ChromaDB Vector Store**: Local, persistent vector database for fast semantic search
- **Intelligent Reranking**: Gemini-powered reranking for improved retrieval quality
- **Web Grounding**: Optional real-time web search to supplement answers with current information
- **Job Tracking**: Background processing with CSV-based job tracking and status monitoring
- **REST API**: FastAPI-based API for all operations
- **Streamlit UI**: User-friendly web interface for querying

### Advanced Features
- Automatic link content fetching (websites and PDFs)
- Enriched document indexing with linked content
- Multi-stage RAG pipeline: Retrieve → Rerank → Generate
- Configurable retrieval parameters
- Source attribution and citation
- Batch embedding with progress tracking

## 📋 Prerequisites

- Python 3.8+
- Google API Key (for Gemini API access)

## 🔧 Installation

1. **Clone or download the repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from: https://aistudio.google.com/apikey

## 🎯 Quick Start

### 1. Start the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 2. Or Use the Streamlit UI

```bash
streamlit run streamlit_app.py
```

The UI will open in your browser at `http://localhost:8501`

## 📚 Usage Guide

### API Workflow

#### Step 1: Upload Documents
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@document.txt" \
  -F "collection=my_collection"
```

#### Step 2: Generate Embeddings
```bash
curl -X POST "http://localhost:8000/api/embed?collection=my_collection"
```

Returns a `job_id` for tracking progress.

#### Step 3: Check Job Status
```bash
curl "http://localhost:8000/api/embed/status/{job_id}"
```

#### Step 4: Query the System
```bash
curl -X POST "http://localhost:8000/api/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "collection": "my_collection",
    "use_grounding": false
  }'
```

## 🔌 API Endpoints

### Document Management
- `POST /api/upload` - Upload a text document
- `GET /api/files` - List uploaded files
- `GET /api/collections` - List vector database collections

### Embedding Pipeline
- `POST /api/embed` - Generate embeddings for a collection (async)
- `GET /api/embed/status/{job_id}` - Check embedding job status
- `GET /api/embed/jobs` - List all embedding jobs
- `GET /api/embed/stats` - Get job statistics
- `DELETE /api/embed/job/{job_id}` - Delete a specific job
- `POST /api/embed/jobs/clear` - Clear completed jobs

### RAG Query Pipeline
- `POST /api/search` - Semantic search without generation
- `POST /api/rag` - Complete RAG pipeline (retrieve, rerank, generate)

### RAG Query Parameters

```json
{
  "query": "Your question",
  "collection": "collection_name",
  "top_k": 10,              // Optional: Final context chunks (default: 10)
  "retrieve_k": 100,        // Optional: Initial retrieval (default: 100)
  "rerank_k": 50,           // Optional: Chunks to rerank (default: 50)
  "use_reranking": true,    // Optional: Enable reranking (default: true)
  "temperature": 0.3,       // Optional: LLM temperature (default: 0.3)
  "use_grounding": false    // Optional: Enable web search (default: false)
}
```

## 🏗️ Architecture

### Components

1. **Main API (`main.py`)**: FastAPI application with all endpoints
2. **Embedder (`embedder.py`)**: Gemini embedding generation and ChromaDB integration
3. **Link Processor (`link_processor.py`)**: Extracts URLs and fetches content from websites/PDFs
4. **Chunker (`chunker.py`)**: Intelligent text chunking with overlap
5. **Reranker (`reranker.py`)**: Gemini-based reranking for improved relevance
6. **PDF Converter (`pdf_converter.py`)**: Extract text from PDF files
7. **Job Tracker (`job_tracker.py`)**: Background job management with CSV persistence
8. **Streamlit UI (`streamlit_app.py`)**: Web interface for querying

### RAG Pipeline

```
Document Upload
    ↓
Link Extraction & Content Fetching
    ↓
Text Chunking (512 chars, 50 overlap)
    ↓
Gemini Embeddings (3072 dim)
    ↓
ChromaDB Vector Store
    ↓
Query → Retrieve (top 100) → Rerank (top 50) → Generate (top 10)
    ↓
Answer + Sources + Citations
```

## 🎨 Key Features Explained

### Link Processing
When you upload a document containing URLs, the system:
1. Extracts all URLs (websites and PDFs)
2. Fetches content from each URL
3. Creates enriched documents with linked content
4. Indexes everything together for comprehensive search

### Smart Defaults
The RAG pipeline uses optimized defaults:
- **Retrieve**: 100 chunks (wide net for good recall)
- **Rerank**: 50 chunks (balance quality vs speed)
- **Generate**: 10 chunks (optimal context for LLM)
- **Temperature**: 0.3 (factual, consistent answers)

### Web Grounding
Enable `use_grounding: true` to supplement answers with real-time web search:
- Gemini automatically searches the web when needed
- Returns web sources and search queries used
- Best for current events or time-sensitive information

## 📁 Project Structure

```
rag_p/
├── main.py                 # FastAPI application
├── embedder.py            # Gemini embeddings & ChromaDB
├── link_processor.py      # URL extraction & content fetching
├── chunker.py             # Text chunking
├── reranker.py            # Document reranking
├── pdf_converter.py       # PDF text extraction
├── job_tracker.py         # Job management
├── streamlit_app.py       # Web UI
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
├── chroma_db/            # Vector database (auto-created)
├── uploaded_files/       # Uploaded documents (auto-created)
├── enriched_files/       # Documents with linked content (auto-created)
├── downloaded_links/     # Individual link content (auto-created)
└── embedding_jobs.csv    # Job tracking (auto-created)
```

## 🔐 Security

- API keys are loaded from environment variables
- `.env` file is excluded from git via `.gitignore`
- No hardcoded credentials in code
- Local-only vector database (no external data transmission)

## 🛠️ Configuration

### Environment Variables
```bash
GOOGLE_API_KEY=your_api_key_here
```

### Link Processing
In `main.py`, configure link processing behavior:
```python
DOWNLOAD_WEBSITES = False  # Gemini grounding handles web content
DOWNLOAD_PDFS = True       # Keep PDFs as they may not be accessible via grounding
```

### Chunking
Adjust in `main.py`:
```python
chunker = TextChunker(chunk_size=512, chunk_overlap=50)
```

### Embedding Dimensions
Change in `embedder.py`:
```python
GeminiEmbedder(api_key=GOOGLE_API_KEY, dimension=3072)
```

## 📊 Example Usage

### Using the API

```python
import requests

# Upload document
files = {'file': open('document.txt', 'rb')}
response = requests.post(
    'http://localhost:8000/api/upload',
    files=files,
    data={'collection': 'my_docs'}
)

# Start embedding
response = requests.post(
    'http://localhost:8000/api/embed?collection=my_docs'
)
job_id = response.json()['job_id']

# Query
response = requests.post(
    'http://localhost:8000/api/rag',
    json={
        'query': 'What are the main findings?',
        'collection': 'my_docs'
    }
)
print(response.json()['answer'])
```

## 🤝 Contributing

This is a standalone RAG system. Feel free to customize and extend it for your needs.

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for embeddings and generation
- ChromaDB for vector storage
- Jina AI for reranking models
- FastAPI for the web framework
- Streamlit for the UI

## 📞 Support

For issues related to:
- **Gemini API**: https://ai.google.dev/docs
- **ChromaDB**: https://docs.trychroma.com/

---

**Built with ❤️ using Google Gemini and ChromaDB**

