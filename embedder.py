import os
from typing import List, Dict, Optional
import numpy as np
from google import genai
from google.genai import types
import chromadb
from chromadb.config import Settings
from pathlib import Path
import uuid
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class GeminiEmbedder:
    """
    Generates embeddings using Google's Gemini embedding model
    Uses gemini-embedding-001 with configurable dimensions (up to 3072)
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gemini-embedding-001",
        dimension: int = 3072
    ):
        """
        Initialize Gemini Embedder with the new genai.Client
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model: Embedding model name (default: gemini-embedding-001)
            dimension: Output embedding dimension (default: 3072)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not provided. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize client with API key
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.dimension = dimension
    
    async def embed_documents(
        self, 
        texts: List[str],
        batch_size: int = 50
    ) -> List[List[float]]:
        """
        Batch embed documents
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        total_texts = len(texts)
        num_batches = (total_texts + batch_size - 1) // batch_size
        
        logging.info(f"Starting embedding generation for {total_texts} documents in {num_batches} batches")
        start_time = time.time()
        
        for batch_idx in range(0, len(texts), batch_size):
            batch = texts[batch_idx:batch_idx + batch_size]
            current_batch = (batch_idx // batch_size) + 1
            
            try:
                # Embed each text in the batch
                batch_embeddings = []
                batch_start = time.time()
                
                for idx, text in enumerate(batch):
                    result = self.client.models.embed_content(
                        model=self.model,
                        contents=text,
                        config=types.EmbedContentConfig(output_dimensionality=self.dimension)
                    )
                    # Extract embedding from result
                    embedding_obj = result.embeddings[0]
                    batch_embeddings.append(embedding_obj.values)
                
                embeddings.extend(batch_embeddings)
                
                batch_time = time.time() - batch_start
                progress = (current_batch / num_batches) * 100
                docs_processed = min(batch_idx + batch_size, total_texts)
                
                logging.info(
                    f"Batch {current_batch}/{num_batches} completed "
                    f"({docs_processed}/{total_texts} docs, {progress:.1f}%) "
                    f"- Time: {batch_time:.2f}s"
                )
                
            except Exception as e:
                logging.error(f"Error embedding batch {current_batch}/{num_batches}: {e}")
                raise
        
        total_time = time.time() - start_time
        avg_time = total_time / total_texts
        logging.info(
            f"✓ Embedding generation completed: {total_texts} documents in {total_time:.2f}s "
            f"(avg: {avg_time:.3f}s per doc)"
        )
        
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query
        
        Args:
            query: Query text to embed
        
        Returns:
            Embedding vector
        """
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=query,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension)
            )
            # Extract embedding from result
            embedding_obj = result.embeddings[0]
            return embedding_obj.values
        except Exception as e:
            print(f"Error embedding query: {e}")
            raise
    
    def embed_documents_sync(
        self, 
        texts: List[str],
        batch_size: int = 50
    ) -> List[List[float]]:
        """Synchronous version of embed_documents"""
        embeddings = []
        total_texts = len(texts)
        num_batches = (total_texts + batch_size - 1) // batch_size
        
        logging.info(f"Starting embedding generation for {total_texts} documents in {num_batches} batches")
        start_time = time.time()
        
        for batch_idx in range(0, len(texts), batch_size):
            batch = texts[batch_idx:batch_idx + batch_size]
            current_batch = (batch_idx // batch_size) + 1
            
            try:
                batch_embeddings = []
                batch_start = time.time()
                
                for text in batch:
                    result = self.client.models.embed_content(
                        model=self.model,
                        contents=text,
                        config=types.EmbedContentConfig(output_dimensionality=self.dimension)
                    )
                    # Extract embedding from result
                    embedding_obj = result.embeddings[0]
                    batch_embeddings.append(embedding_obj.values)
                
                embeddings.extend(batch_embeddings)
                
                batch_time = time.time() - batch_start
                progress = (current_batch / num_batches) * 100
                docs_processed = min(batch_idx + batch_size, total_texts)
                
                logging.info(
                    f"Batch {current_batch}/{num_batches} completed "
                    f"({docs_processed}/{total_texts} docs, {progress:.1f}%) "
                    f"- Time: {batch_time:.2f}s"
                )
                
            except Exception as e:
                logging.error(f"Error embedding batch {current_batch}/{num_batches}: {e}")
                raise
        
        total_time = time.time() - start_time
        avg_time = total_time / total_texts
        logging.info(
            f"✓ Embedding generation completed: {total_texts} documents in {total_time:.2f}s "
            f"(avg: {avg_time:.3f}s per doc)"
        )
        
        return embeddings
    
    def embed_query_sync(self, query: str) -> List[float]:
        """Synchronous version of embed_query"""
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=query,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension)
            )
            # Extract embedding from result
            embedding_obj = result.embeddings[0]
            return embedding_obj.values
        except Exception as e:
            print(f"Error embedding query: {e}")
            raise


class ChromaDBVectorStore:
    """
    Local vector database using ChromaDB
    Stores embeddings with metadata for fast retrieval
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
    
    def create_collection(
        self, 
        name: str,
        dimension: int = 768,
        metadata: Optional[Dict] = None
    ):
        """
        Create or get a collection (partition)
        
        Args:
            name: Collection name
            dimension: Embedding dimension
            metadata: Optional metadata for the collection
        
        Returns:
            ChromaDB collection object
        """
        collection_metadata = metadata or {
            "hnsw:space": "cosine",
            "dimension": dimension
        }
        
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata=collection_metadata
            )
            return collection
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
    def add_documents(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents with embeddings to a collection
        
        Args:
            collection_name: Name of the collection
            texts: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs (auto-generated if not provided)
        """
        try:
            logging.info(f"Adding {len(texts)} documents to collection '{collection_name}'")
            start_time = time.time()
            
            collection = self.client.get_collection(collection_name)
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]
                logging.info(f"Generated {len(ids)} unique document IDs")
            
            # Generate empty metadata if not provided
            if metadatas is None:
                metadatas = [{} for _ in range(len(texts))]
            
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            elapsed_time = time.time() - start_time
            logging.info(
                f"✓ Successfully added {len(texts)} documents to '{collection_name}' "
                f"in {elapsed_time:.2f}s"
            )
            
            return ids
        except Exception as e:
            logging.error(f"Error adding documents to collection '{collection_name}': {e}")
            raise
    
    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Vector similarity search
        
        Args:
            collection_name: Name of the collection to search
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Optional metadata filter
        
        Returns:
            Dictionary with ids, documents, distances, and metadatas
        """
        try:
            collection = self.client.get_collection(collection_name)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
            
            return {
                "ids": results['ids'][0],
                "documents": results['documents'][0],
                "distances": results['distances'][0],
                "metadatas": results['metadatas'][0]
            }
        except Exception as e:
            print(f"Error searching: {e}")
            raise
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
        except Exception as e:
            print(f"Error deleting collection: {e}")
            raise
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            raise
    
    def get_collection_count(self, collection_name: str) -> int:
        """Get number of documents in a collection"""
        try:
            collection = self.client.get_collection(collection_name)
            return collection.count()
        except Exception as e:
            print(f"Error getting collection count: {e}")
            raise


# Convenience function to create embedder and vector store
def create_embedding_pipeline(
    api_key: Optional[str] = None,
    persist_directory: str = "./chroma_db"
) -> tuple[GeminiEmbedder, ChromaDBVectorStore]:
    """
    Create embedder and vector store instances
    
    Args:
        api_key: Google API key
        persist_directory: ChromaDB storage directory
    
    Returns:
        Tuple of (embedder, vector_store)
    """
    embedder = GeminiEmbedder(api_key=api_key)
    
    vector_store = ChromaDBVectorStore(
        persist_directory=persist_directory
    )
    
    return embedder, vector_store


# Example usage
if __name__ == "__main__":
    # Initialize with gemini-embedding-001 (3072 dimensions)
    embedder, vector_store = create_embedding_pipeline()
    
    # Create a collection
    collection_name = "test_collection"
    vector_store.create_collection(collection_name, dimension=3072)
    
    # Example documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language."
    ]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embedder.embed_documents_sync(documents)
    print(f"Generated {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")
    
    # Add to vector store
    doc_ids = vector_store.add_documents(
        collection_name=collection_name,
        texts=documents,
        embeddings=embeddings
    )
    
    print(f"Added {len(doc_ids)} documents to collection '{collection_name}'")
    
    # Search example
    query = "What is deep learning?"
    print(f"\nSearching for: '{query}'")
    query_embedding = embedder.embed_query_sync(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    results = vector_store.search(collection_name, query_embedding, top_k=2)
    
    print(f"\nTop {len(results['documents'])} search results:")
    for i, (doc, distance) in enumerate(zip(results['documents'], results['distances'])):
        print(f"{i+1}. {doc}")
        print(f"   Distance: {distance:.4f}\n")

