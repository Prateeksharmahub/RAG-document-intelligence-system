"""
Document Reranking Module
Uses state-of-the-art reranking models to improve retrieval quality
"""

from typing import List, Dict, Tuple, Optional
from sentence_transformers import CrossEncoder
import numpy as np
import re
from google import genai
from google.genai import types


class DocumentReranker:
    """
    Rerank retrieved documents using cross-encoder models
    Supports multiple reranking strategies
    """
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        device: str = "cpu"
    ):
        """
        Initialize reranker
        
        Args:
            model_name: HuggingFace model name
                - jinaai/jina-reranker-v2-base-multilingual (Best overall, 512M params)
                - BAAI/bge-reranker-v2-m3 (Good, multilingual)
                - BAAI/bge-reranker-base (Fast, English)
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (Fastest, English)
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(
            model_name, 
            max_length=512, 
            device=device,
            trust_remote_code=True  # Required for Jina models
        )
        print(f"Reranker loaded successfully on {device}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Dict]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top documents to return (None = all)
        
        Returns:
            List of reranked documents with scores
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        ranked_indices = np.argsort(scores)[::-1]
        
        # Return top_k results
        if top_k:
            ranked_indices = ranked_indices[:top_k]
        
        results = []
        for idx in ranked_indices:
            results.append({
                'index': int(idx),
                'score': float(scores[idx]),
                'text': documents[idx]
            })
        
        return results
    
    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None,
        text_key: str = 'text'
    ) -> List[Dict]:
        """
        Rerank documents with metadata
        
        Args:
            query: Search query
            documents: List of document dicts with text and metadata
            top_k: Number of top documents to return
            text_key: Key for text content in document dict
        
        Returns:
            List of reranked documents with scores and metadata
        """
        if not documents:
            return []
        
        # Extract texts
        texts = [doc[text_key] for doc in documents]
        
        # Create query-document pairs
        pairs = [[query, text] for text in texts]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        ranked_indices = np.argsort(scores)[::-1]
        
        # Return top_k results
        if top_k:
            ranked_indices = ranked_indices[:top_k]
        
        results = []
        for idx in ranked_indices:
            reranked_doc = documents[idx].copy()
            reranked_doc['rerank_score'] = float(scores[idx])
            reranked_doc['original_rank'] = int(idx)
            results.append(reranked_doc)
        
        return results
    
    def batch_rerank(
        self,
        queries: List[str],
        document_lists: List[List[str]],
        top_k: int = None
    ) -> List[List[Dict]]:
        """
        Rerank multiple query-document sets in batch
        
        Args:
            queries: List of queries
            document_lists: List of document lists (one per query)
            top_k: Number of top documents per query
        
        Returns:
            List of reranked results (one per query)
        """
        results = []
        for query, docs in zip(queries, document_lists):
            reranked = self.rerank(query, docs, top_k)
            results.append(reranked)
        
        return results


class GeminiReranker:
    """
    LLM-based reranker using Gemini (like RankGPT)
    Uses Gemini to read and rank documents by relevance
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro",
        max_docs_per_call: int = 20
    ):
        """
        Initialize Gemini reranker
        
        Args:
            api_key: Google API key
            model: Gemini model to use (default: gemini-2.0-flash-exp for speed)
            max_docs_per_call: Max documents to rank in one call
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_docs_per_call = max_docs_per_call
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Dict]:
        """
        Rerank documents using Gemini LLM
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top documents to return
        
        Returns:
            List of reranked documents with scores
        """
        if not documents:
            return []
        
        # Limit number of documents to avoid context limits
        docs_to_rank = documents[:self.max_docs_per_call]
        
        # Create prompt for ranking
        prompt = self._create_ranking_prompt(query, docs_to_rank)
        
        try:
            # Get ranking from Gemini
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for consistent ranking
                    max_output_tokens=500
                )
            )
            
            # Parse ranking from response
            ranked_indices = self._parse_ranking(response.text, len(docs_to_rank))
            
            # Create results
            results = []
            for rank, idx in enumerate(ranked_indices):
                if idx < len(docs_to_rank):
                    results.append({
                        'index': idx,
                        'score': 1.0 - (rank / len(ranked_indices)),  # Score decreases with rank
                        'text': docs_to_rank[idx],
                        'rank': rank + 1
                    })
            
            # Return top_k if specified
            if top_k:
                results = results[:top_k]
            
            return results
        
        except Exception as e:
            print(f"Error in Gemini reranking: {e}")
            # Fallback: return original order with decreasing scores
            return [
                {
                    'index': i,
                    'score': 1.0 - (i / len(docs_to_rank)),
                    'text': doc,
                    'rank': i + 1
                }
                for i, doc in enumerate(docs_to_rank[:top_k] if top_k else docs_to_rank)
            ]
    
    def _create_ranking_prompt(self, query: str, documents: List[str]) -> str:
        """Create prompt for LLM ranking"""
        docs_text = ""
        for i, doc in enumerate(documents):
            # Truncate long documents
            doc_preview = doc[:300] + "..." if len(doc) > 300 else doc
            docs_text += f"[{i+1}] {doc_preview}\n\n"
        
        prompt = f"""You are an expert at ranking document relevance for search queries.

Query: {query}

Documents to rank:
{docs_text}

Task: Rank these documents from most relevant to least relevant for answering the query.
Output ONLY the ranking as a comma-separated list of numbers, like: 3,1,5,2,4

Ranking (most relevant first):"""
        
        return prompt
    
    def _parse_ranking(self, response: str, num_docs: int) -> List[int]:
        """Parse ranking from LLM response"""
        try:
            # Extract numbers from response
            numbers = re.findall(r'\d+', response)
            ranking = [int(n) - 1 for n in numbers if 1 <= int(n) <= num_docs]  # Convert to 0-indexed
            
            # Add any missing documents at the end
            missing = [i for i in range(num_docs) if i not in ranking]
            ranking.extend(missing)
            
            return ranking[:num_docs]
        except:
            # Fallback: return original order
            return list(range(num_docs))
    
    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None,
        text_key: str = 'text'
    ) -> List[Dict]:
        """
        Rerank documents with metadata
        
        Args:
            query: Search query
            documents: List of document dicts
            top_k: Number of top documents to return
            text_key: Key for text in document dict
        
        Returns:
            List of reranked documents with metadata
        """
        if not documents:
            return []
        
        # Extract texts
        texts = [doc[text_key] for doc in documents]
        
        # Rerank
        reranked = self.rerank(query, texts, top_k=len(documents))
        
        # Map back to original documents
        results = []
        for item in reranked[:top_k] if top_k else reranked:
            orig_doc = documents[item['index']].copy()
            orig_doc['rerank_score'] = item['score']
            orig_doc['original_rank'] = item['index'] + 1
            orig_doc['new_rank'] = item['rank']
            results.append(orig_doc)
        
        return results


# Factory function for easy initialization
def create_reranker(
    model_type: str = "gemini",
    device: str = "cpu",
    api_key: str = None
) -> DocumentReranker:
    """
    Create a reranker with preset configurations
    
    Args:
        model_type: 'gemini' (best quality), 'jina', 'bge', 'minilm'
        device: 'cpu' or 'cuda' (for cross-encoder models)
        api_key: Required for 'gemini'
    
    Returns:
        Initialized Reranker (DocumentReranker or GeminiReranker)
    """
    if model_type == 'gemini':
        if not api_key:
            raise ValueError("api_key required for Gemini reranker")
        return GeminiReranker(api_key=api_key)
    
    # Cross-encoder models
    models = {
        'jina': 'jinaai/jina-reranker-v2-base-multilingual',
        'bge': 'BAAI/bge-reranker-v2-m3',
        'bge-base': 'BAAI/bge-reranker-base',
        'minilm': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    }
    
    model_name = models.get(model_type, models['jina'])
    return DocumentReranker(model_name=model_name, device=device)


# Example usage
if __name__ == "__main__":
    # Initialize reranker
    reranker = create_reranker(model_type='jina', device='cpu')
    
    # Example query and documents
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on data.",
        "Python is a popular programming language.",
        "Deep learning uses neural networks with multiple layers.",
        "The weather today is sunny and warm.",
        "Machine learning algorithms learn from data to make predictions."
    ]
    
    # Rerank documents
    results = reranker.rerank(query, documents, top_k=3)
    
    print(f"Query: {query}\n")
    print("Top 3 reranked results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result['score']:.4f}")
        print(f"   Text: {result['text']}")

