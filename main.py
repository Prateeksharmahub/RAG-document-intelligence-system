import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable ChromaDB telemetry warnings
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=4)

# Import our custom modules
from embedder import GeminiEmbedder, ChromaDBVectorStore
from chunker import TextChunker
from job_tracker import JobTracker
from reranker import DocumentReranker, create_reranker
from link_processor import LinkProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Custom RAG API",
    description="RAG System with Gemini Embeddings and Local Vector DB",
    version="1.0.0"
)

# Configuration
UPLOAD_DIR = Path("./uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

# Directory for enriched files (with link content)
ENRICHED_DIR = Path("./enriched_files")
ENRICHED_DIR.mkdir(exist_ok=True)

# Directory for individual downloaded links
LINKS_DIR = Path("./downloaded_links")
LINKS_DIR.mkdir(exist_ok=True)

# Supported file formats
SUPPORTED_FORMATS = ["txt"]

# Link processing configuration
# Set to False since Gemini handles grounding for web content
DOWNLOAD_WEBSITES = False  # Gemini grounding handles web content
DOWNLOAD_PDFS = True  # Keep PDFs as they may not be accessible via grounding

# Load Google API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables. "
        "Please create a .env file with your Google API key or set it as an environment variable."
    )

# Initialize embedder and vector store
embedder = GeminiEmbedder(api_key=GOOGLE_API_KEY)
vector_store = ChromaDBVectorStore(persist_directory="./chroma_db")
chunker = TextChunker(chunk_size=512, chunk_overlap=50)
link_processor = LinkProcessor(
    download_websites=DOWNLOAD_WEBSITES,
    download_pdfs=DOWNLOAD_PDFS
)

# Initialize reranker (lazy loading - will load when first used)
reranker = None

# Job tracking with CSV persistence
job_tracker = JobTracker(csv_file="./embedding_jobs.csv")


def get_reranker():
    """Lazy load reranker on first use"""
    global reranker
    if reranker is None:
        print("Loading Gemini reranker (first time only)...")
        reranker = create_reranker(model_type='gemini', api_key=GOOGLE_API_KEY)
    return reranker


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Custom RAG API",
        "version": "1.0.0"
    }


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = "default"
):
    """
    Upload a TXT document and store it locally
    
    Args:
        file: TXT file to upload
        collection: Collection/partition name (default: "default")
    
    Returns:
        JSON response with file_id, filename, and file_path
    """
    try:
        # Validate file format
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Only {SUPPORTED_FORMATS} are supported."
            )
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Create collection directory if it doesn't exist
        collection_dir = UPLOAD_DIR / collection
        collection_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file_id[:8]}_{file.filename}"
        file_path = collection_dir / safe_filename
        
        # Save file to disk
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully",
                "file_id": file_id,
                "filename": file.filename,
                "safe_filename": safe_filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "collection": collection,
                "upload_date": datetime.now().isoformat(),
                "status": "uploaded"
            }
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )


@app.get("/api/files")
async def list_files(collection: str = None):
    """
    List all uploaded files
    
    Args:
        collection: Filter by collection name (optional)
    
    Returns:
        List of uploaded files
    """
    try:
        files_list = []
        
        if collection:
            # List files in specific collection
            collection_dir = UPLOAD_DIR / collection
            if collection_dir.exists():
                for file_path in collection_dir.glob("*.*"):
                    files_list.append({
                        "filename": file_path.name,
                        "file_path": str(file_path),
                        "file_size": os.path.getsize(file_path),
                        "collection": collection,
                        "modified_date": datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat()
                    })
        else:
            # List all files in all collections
            for collection_dir in UPLOAD_DIR.iterdir():
                if collection_dir.is_dir():
                    for file_path in collection_dir.glob("*.*"):
                        files_list.append({
                            "filename": file_path.name,
                            "file_path": str(file_path),
                            "file_size": os.path.getsize(file_path),
                            "collection": collection_dir.name,
                            "modified_date": datetime.fromtimestamp(
                                os.path.getmtime(file_path)
                            ).isoformat()
                        })
        
        return {
            "count": len(files_list),
            "files": files_list
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing files: {str(e)}"
        )


def process_embeddings_sync(job_id: str, collection_name: str):
    """
    Synchronous background task to process embeddings for a collection
    Runs in a separate thread to avoid blocking the main API
    """
    try:
        # Update job status
        job_tracker.update_job(job_id, status="processing", message="Reading files...")
        
        # Find all text files in the collection
        collection_dir = UPLOAD_DIR / collection_name
        if not collection_dir.exists():
            job_tracker.update_job(
                job_id,
                status="failed",
                message=f"Collection '{collection_name}' not found"
            )
            return
        
        txt_files = list(collection_dir.glob("*.txt"))
        if not txt_files:
            job_tracker.update_job(
                job_id,
                status="failed",
                message=f"No .txt files found in collection '{collection_name}'"
            )
            return
        
        job_tracker.update_job(job_id, total_files=len(txt_files))
        
        # Create or get the collection in vector store
        vector_store.create_collection(collection_name, dimension=3072)
        
        all_chunks = []
        all_metadatas = []
        processed_files = 0
        
        # Process each file
        for txt_file in txt_files:
            job_tracker.update_job(job_id, message=f"Processing {txt_file.name}...")
            
            # Read file content
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process links in the document (fetch and enrich content)
            job_tracker.update_job(job_id, message=f"Processing links in {txt_file.name}...")
            link_result = link_processor.process_document_with_links(content)
            
            # Use enriched content if links were found and fetched
            if link_result['urls_fetched'] > 0:
                content_to_chunk = link_result['enriched_text']
                print(f"Enriched {txt_file.name} with {link_result['urls_fetched']} linked sources")
                
                # Save enriched content to file
                job_tracker.update_job(job_id, message=f"Saving enriched content for {txt_file.name}...")
                enriched_collection_dir = ENRICHED_DIR / collection_name
                enriched_collection_dir.mkdir(exist_ok=True)
                
                # Create enriched filename (prefix with "enriched_")
                enriched_filename = f"enriched_{txt_file.name}"
                enriched_path = enriched_collection_dir / enriched_filename
                
                with open(enriched_path, 'w', encoding='utf-8') as f:
                    f.write(content_to_chunk)
                
                print(f"✓ Saved enriched content to: {enriched_path}")
                
                # Save individual downloaded links
                links_collection_dir = LINKS_DIR / collection_name / txt_file.stem
                links_collection_dir.mkdir(parents=True, exist_ok=True)
                
                for idx, link_content in enumerate(link_result['linked_content'], 1):
                    if link_content.get('status') == 'success':
                        # Create safe filename from URL
                        url = link_content['url']
                        safe_url = url.replace('https://', '').replace('http://', '').replace('/', '_').replace(':', '_')
                        if len(safe_url) > 100:
                            safe_url = safe_url[:100]
                        
                        link_type = link_content.get('content_type', 'unknown')
                        link_filename = f"link_{idx}_{link_type}_{safe_url}.txt"
                        link_path = links_collection_dir / link_filename
                        
                        # Write link content with metadata
                        with open(link_path, 'w', encoding='utf-8') as f:
                            f.write(f"URL: {url}\n")
                            f.write(f"Type: {link_type}\n")
                            f.write(f"Title: {link_content.get('title', 'N/A')}\n")
                            f.write(f"Character Count: {link_content.get('char_count', len(link_content.get('text', '')))}\n")
                            if link_content.get('num_pages'):
                                f.write(f"Pages: {link_content['num_pages']}\n")
                            f.write(f"Truncated: {link_content.get('was_truncated', False)}\n")
                            f.write(f"\n{'='*80}\n\n")
                            f.write(link_content.get('text', ''))
                        
                        print(f"  ✓ Saved link {idx} to: {link_path}")
            else:
                content_to_chunk = content
            
            # Chunk the text (now includes linked content if any)
            chunks = chunker.chunk_text(
                content_to_chunk,
                metadata={
                    'source_file': txt_file.name,
                    'collection': collection_name,
                    'file_path': str(txt_file),
                    'has_links': link_result['urls_found'] > 0,
                    'urls_found': link_result['urls_found'],
                    'urls_fetched': link_result['urls_fetched']
                }
            )
            
            all_chunks.extend([chunk['text'] for chunk in chunks])
            all_metadatas.extend([chunk['metadata'] for chunk in chunks])
            
            processed_files += 1
            job_tracker.update_job(job_id, processed_files=processed_files)
        
        job_tracker.update_job(
            job_id,
            total_chunks=len(all_chunks),
            message=f"Generating embeddings for {len(all_chunks)} chunks..."
        )
        
        # Generate embeddings (synchronous version)
        embeddings = embedder.embed_documents_sync(all_chunks, batch_size=50)
        
        job_tracker.update_job(job_id, message="Storing embeddings in vector database...")
        
        # Store in vector database
        doc_ids = vector_store.add_documents(
            collection_name=collection_name,
            texts=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas
        )
        
        # Update job status to completed
        job_tracker.update_job(
            job_id,
            status="completed",
            message="Embeddings generated and stored successfully",
            completed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        job_tracker.update_job(
            job_id,
            status="failed",
            message=f"Error: {str(e)}",
            error=str(e)
        )


@app.post("/api/embed")
async def create_embeddings(collection: str):
    """
    Generate embeddings for all text files in a collection
    Runs in a separate thread to avoid blocking the API
    
    Args:
        collection: Name of the collection to process
    
    Returns:
        Job ID for tracking the embedding process
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job in tracker
        job_tracker.create_job(
            job_id=job_id,
            collection=collection,
            status="queued",
            message="Job queued"
        )
        
        # Submit task to thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, process_embeddings_sync, job_id, collection)
        
        return {
            "job_id": job_id,
            "collection": collection,
            "status": "queued",
            "message": "Embedding job queued successfully. Use /api/embed/status/{job_id} to check progress."
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating embedding job: {str(e)}"
        )


@app.get("/api/embed/status/{job_id}")
async def get_embedding_status(job_id: str):
    """
    Get the status of an embedding job
    
    Args:
        job_id: Job ID returned from /api/embed
    
    Returns:
        Job status and progress information
    """
    job = job_tracker.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job ID '{job_id}' not found"
        )
    
    return job


@app.get("/api/embed/jobs")
async def list_embedding_jobs():
    """
    List all embedding jobs
    
    Returns:
        List of all embedding jobs with their status
    """
    jobs = job_tracker.get_all_jobs()
    stats = job_tracker.get_stats()
    
    return {
        "total_jobs": len(jobs),
        "stats": stats,
        "jobs": jobs
    }


@app.get("/api/embed/stats")
async def get_embedding_stats():
    """
    Get job statistics
    
    Returns:
        Job statistics
    """
    return job_tracker.get_stats()


@app.delete("/api/embed/job/{job_id}")
async def delete_embedding_job(job_id: str):
    """
    Delete an embedding job
    
    Args:
        job_id: Job ID to delete
    
    Returns:
        Success message
    """
    deleted = job_tracker.delete_job(job_id)
    
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Job ID '{job_id}' not found"
        )
    
    return {"message": f"Job {job_id} deleted successfully"}


@app.post("/api/embed/jobs/clear")
async def clear_completed_jobs():
    """
    Clear all completed jobs
    
    Returns:
        Number of jobs cleared
    """
    count = job_tracker.clear_completed_jobs()
    return {
        "message": f"Cleared {count} completed jobs",
        "count": count
    }


@app.get("/api/embed/export")
async def export_jobs_csv():
    """
    Export jobs to CSV file
    
    Returns:
        Success message with file path
    """
    output_file = f"jobs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    success = job_tracker.export_to_csv(output_file)
    
    if success:
        return {
            "message": "Jobs exported successfully",
            "file": output_file
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Error exporting jobs"
        )


@app.get("/api/collections")
async def list_collections():
    """
    List all vector database collections
    
    Returns:
        List of collection names and their document counts
    """
    try:
        collections = vector_store.list_collections()
        
        collection_info = []
        for col_name in collections:
            count = vector_store.get_collection_count(col_name)
            collection_info.append({
                "name": col_name,
                "document_count": count
            })
        
        return {
            "total_collections": len(collections),
            "collections": collection_info
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing collections: {str(e)}"
        )


# ==================== PHASE 2: RAG QUERY PIPELINE ====================

@app.post("/api/search")
async def search_documents(
    query: str,
    collection: str,
    top_k: int = 10
):
    """
    Search for similar documents in a collection
    
    Args:
        query: Search query text
        collection: Collection name to search in
        top_k: Number of results to return (default: 10)
    
    Returns:
        List of similar documents with scores and metadata
    """
    try:
        # Check if collection exists
        collections = vector_store.list_collections()
        if collection not in collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection}' not found"
            )
        
        # Generate query embedding
        query_embedding = embedder.embed_query_sync(query)
        
        # Search vector database
        results = vector_store.search(
            collection_name=collection,
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'])):
            formatted_results.append({
                "rank": i + 1,
                "document_id": results['ids'][i],
                "text": results['documents'][i],
                "distance": float(results['distances'][i]),
                "similarity_score": 1 - float(results['distances'][i]),  # Convert distance to similarity
                "metadata": results['metadatas'][i]
            })
        
        return {
            "query": query,
            "collection": collection,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )


@app.post("/api/rag")
async def rag_query(
    query: str,
    collection: str,
    top_k: Optional[int] = None,
    retrieve_k: Optional[int] = None,
    rerank_k: Optional[int] = None,
    use_reranking: Optional[bool] = None,
    temperature: Optional[float] = None,
    use_grounding: bool = False
):
    """
    Complete RAG pipeline: Retrieve, Rerank (optional), Generate
    
    Args:
        query: User question (required)
        collection: Collection to search (required)
        top_k: Number of final context chunks to use for generation (optional, auto: 5-10 based on reranking)
        retrieve_k: Number of chunks to retrieve initially (optional, auto: optimized based on reranking)
        rerank_k: Number of chunks to rerank (optional, auto: optimized subset of retrieved chunks)
        use_reranking: Whether to apply reranking (optional, auto: True for better quality)
        temperature: LLM temperature for generation (optional, auto: 0.3 for factual answers)
        use_grounding: Whether to enable Gemini web search grounding (default: False)
    
    Returns:
        Generated answer with sources and metadata
    
    Smart Defaults (when not specified):
        - use_reranking: True (enables reranking for better results)
        - top_k: 10 (optimal context size for LLM)
        - retrieve_k: 100 (cast a wide net for initial retrieval)
        - rerank_k: 50 (rerank top 50 for efficiency)
        - temperature: 0.3 (lower temp for factual, grounded answers)
    
    Example:
        Minimal request: {"query": "...", "collection": "..."}
        -> Uses: retrieve_k=100 → rerank 50 → generate with top 10
    """
    try:
        # ==================== SMART DEFAULTS ====================
        # Apply intelligent defaults based on best practices
        
        # Default: Enable reranking for better quality
        if use_reranking is None:
            use_reranking = True
        
        # Default: 10 chunks provides good context without overwhelming the LLM
        if top_k is None:
            top_k = 10
        
        # Default: Retrieve 100 chunks initially for good recall
        # If user disabled reranking, retrieve fewer to save time
        if retrieve_k is None:
            retrieve_k = 100 if use_reranking else 20
        
        # Default: Rerank top 50 chunks (good balance of quality vs speed)
        if rerank_k is None:
            rerank_k = 50 if use_reranking else retrieve_k
        
        # Default: Lower temperature (0.3) for more factual, consistent answers
        # Higher temps (0.7-1.0) are better for creative tasks
        if temperature is None:
            temperature = 0.3
        
        # ==================== END SMART DEFAULTS ====================
        
        # Check if collection exists
        collections = vector_store.list_collections()
        if collection not in collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection}' not found"
            )
        
        # Step 1: Retrieve documents
        # Ensure logical order: retrieve_k >= rerank_k >= top_k
        retrieve_k = max(retrieve_k, top_k)
        rerank_k = min(rerank_k, retrieve_k)
        rerank_k = max(rerank_k, top_k)
        
        query_embedding = embedder.embed_query_sync(query)
        results = vector_store.search(
            collection_name=collection,
            query_embedding=query_embedding,
            top_k=retrieve_k
        )
        
        # Step 2: Rerank using Jina reranker (if enabled)
        if use_reranking:
            # Take top rerank_k documents for reranking
            docs_for_rerank = []
            num_to_rerank = min(rerank_k, len(results['documents']))
            
            for i in range(num_to_rerank):
                docs_for_rerank.append({
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'original_score': 1 - float(results['distances'][i]),
                    'original_rank': i + 1
                })
            
            # Use Jina reranker for proper cross-encoder reranking
            reranker_model = get_reranker()
            ranked_results = reranker_model.rerank_with_metadata(
                query=query,
                documents=docs_for_rerank,
                top_k=top_k,
                text_key='text'
            )
            
            # Update score field for consistency
            for result in ranked_results:
                result['score'] = result['rerank_score']
        else:
            # No reranking - use original retrieval scores
            ranked_results = [
                {
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'score': 1 - float(results['distances'][i]),
                    'original_rank': i + 1
                }
                for i in range(min(top_k, len(results['documents'])))
            ]
        
        # Step 3: Generate answer using LLM (Gemini)
        # Format context
        context_parts = []
        for i, result in enumerate(ranked_results):
            source_file = result['metadata'].get('source_file', 'Unknown')
            chunk_index = result['metadata'].get('chunk_index', 0)
            context_parts.append(
                f"[Source {i+1}] (File: {source_file}, Chunk: {chunk_index})\n{result['text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using Gemini
        from google.genai import types
        
        # Adjust prompt based on whether grounding is enabled
        if use_grounding:
            prompt = f"""You are a helpful AI assistant with access to web search. Answer the user's question using both the provided context from our knowledge base AND current information from the web.

Context from Knowledge Base:
{context}

Question: {query}

Instructions:
- Provide a clear, well-written answer that synthesizes information from all available sources
- Write naturally and conversationally - avoid fragmented or repetitive statements
- If the question requires current information, recent data, or real-time facts, use web search to supplement your answer
- Only cite sources [Source N] for specific facts, data points, or quotes - don't cite after every sentence
- If using web search, mention it naturally (e.g., "Recent reports indicate..." or "According to current information...")
- If information is not available, state it clearly once
- Be accurate and don't make up information

Answer:"""
        else:
            prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Provide a clear, well-written answer that synthesizes the information naturally
- Write in a flowing, conversational style - avoid fragmented or repetitive statements
- Only cite sources [Source N] for specific facts, data points, or direct quotes - not after every sentence
- If the answer cannot be found in the context, state it clearly once
- Be accurate and don't make up information

Answer:"""

        answer = None
        grounding_metadata = None
        try:
            # Use Gemini for generation
            print(f"\n{'='*60}")
            print("GENERATING ANSWER WITH GEMINI")
            print(f"{'='*60}")
            print(f"Model: gemini-2.5-pro")
            print(f"Temperature: {temperature}")
            print(f"Context length: {len(context)} chars")
            print(f"Web Grounding: {'Enabled' if use_grounding else 'Disabled'}")
            print(f"{'='*60}\n")
            
            # Configure generation settings
            # Add grounding tools if enabled (tools go inside the config)
            tools = None
            if use_grounding:
                tools = [types.Tool(google_search=types.GoogleSearch())]
            
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=10000,
                tools=tools
            )
            
            response = embedder.client.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt,
                config=config
            )
            
            # Print raw response for debugging
            print(f"\n{'='*60}")
            print("RAW GEMINI RESPONSE")
            print(f"{'='*60}")
            print(f"Response type: {type(response)}")
            print(f"Response attributes: {dir(response)}")
            print(f"Response: {response}")
            print(f"{'='*60}\n")
            
            # Check if response has text
            if hasattr(response, 'text') and response.text:
                answer = response.text
                print(f"✓ Got answer via response.text ({len(answer)} chars)")
            else:
                # Try alternative attribute
                if hasattr(response, 'candidates') and response.candidates:
                    answer = response.candidates[0].content.parts[0].text
                    print(f"✓ Got answer via candidates[0] ({len(answer)} chars)")
                else:
                    answer = "[No response generated from LLM]"
                    print("✗ No text found in response")
            
            # Extract grounding metadata if available
            if use_grounding and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    grounding_metadata = {
                        'grounding_chunks': [],
                        'web_search_queries': []
                    }
                    
                    # Extract grounding chunks (check if not None before iterating)
                    if hasattr(candidate.grounding_metadata, 'grounding_chunks') and candidate.grounding_metadata.grounding_chunks:
                        for chunk in candidate.grounding_metadata.grounding_chunks:
                            chunk_info = {}
                            if hasattr(chunk, 'web'):
                                chunk_info['uri'] = chunk.web.uri if hasattr(chunk.web, 'uri') else None
                                chunk_info['title'] = chunk.web.title if hasattr(chunk.web, 'title') else None
                            grounding_metadata['grounding_chunks'].append(chunk_info)
                    
                    # Extract web search queries (check if not None before converting to list)
                    if hasattr(candidate.grounding_metadata, 'web_search_queries') and candidate.grounding_metadata.web_search_queries:
                        grounding_metadata['web_search_queries'] = list(candidate.grounding_metadata.web_search_queries)
                    
                    print(f"✓ Extracted grounding metadata: {len(grounding_metadata['grounding_chunks'])} web sources")
                    print(f"  Web queries used: {grounding_metadata['web_search_queries']}")
            
        except Exception as llm_error:
            # Fallback: return context without generation
            import traceback
            error_details = traceback.format_exc()
            print(f"LLM Generation Error: {error_details}")
            answer = f"[LLM Generation Error: {str(llm_error)}]\n\nRelevant context found:\n{context[:500]}..."
        
        # Final safety check
        if answer is None:
            answer = f"[Error: No answer generated]\n\nRelevant context:\n{context[:500]}..."
        
        # Format sources
        sources = []
        for i, result in enumerate(ranked_results):
            sources.append({
                "source_number": i + 1,
                "file": result['metadata'].get('source_file', 'Unknown'),
                "chunk_index": result['metadata'].get('chunk_index', 0),
                "similarity_score": result['score'],
                "text_preview": result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            })
        
        response_data = {
            "query": query,
            "answer": answer,
            "collection": collection,
            "sources": sources,
            "num_sources": len(sources),
            "metadata": {
                "retrieve_k": retrieve_k,
                "rerank_k": rerank_k if use_reranking else None,
                "top_k": top_k,
                "use_reranking": use_reranking,
                "use_grounding": use_grounding,
                "temperature": temperature,
                "pipeline": f"Retrieve {retrieve_k} → {'Rerank ' + str(rerank_k) + ' → ' if use_reranking else ''}Generate with {top_k}{' + Web Grounding' if use_grounding else ''}"
            }
        }
        
        # Add grounding metadata if available
        if grounding_metadata:
            response_data["grounding"] = grounding_metadata
        
        return response_data
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in RAG query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

