"""
Text Chunking Module
Intelligent chunking for document processing
"""

from typing import List, Dict
import re


class TextChunker:
    """
    Chunk text into smaller segments for embedding
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separator: Primary separator for splitting (default: double newline)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text into smaller segments
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        if not text or not text.strip():
            return []
        
        # Split by separator first
        splits = text.split(self.separator)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for split in splits:
            split = split.strip()
            if not split:
                continue
            
            # If adding this split would exceed chunk_size, save current chunk
            if len(current_chunk) + len(split) > self.chunk_size and current_chunk:
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata['chunk_index'] = chunk_index
                chunk_metadata['char_count'] = len(current_chunk)
                chunk_metadata['word_count'] = len(current_chunk.split())
                
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': chunk_metadata
                })
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Take last chunk_overlap characters from current chunk
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + split
                else:
                    current_chunk = split
                
                chunk_index += 1
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + split
                else:
                    current_chunk = split
        
        # Add the last chunk
        if current_chunk:
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = chunk_index
            chunk_metadata['char_count'] = len(current_chunk)
            chunk_metadata['word_count'] = len(current_chunk.split())
            
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def chunk_by_sentences(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text by sentences, respecting chunk_size
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences using regex
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk_size, save current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata['chunk_index'] = chunk_index
                chunk_metadata['char_count'] = len(current_chunk)
                chunk_metadata['word_count'] = len(current_chunk.split())
                
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': chunk_metadata
                })
                
                current_chunk = sentence
                chunk_index += 1
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = chunk_index
            chunk_metadata['char_count'] = len(current_chunk)
            chunk_metadata['word_count'] = len(current_chunk.split())
            
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': chunk_metadata
            })
        
        return chunks


# Example usage
if __name__ == "__main__":
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    
    sample_text = """
    Machine learning is a subset of artificial intelligence. It focuses on 
    building systems that can learn from data.
    
    Deep learning is a type of machine learning that uses neural networks with 
    multiple layers. These networks can learn complex patterns in data.
    
    Natural language processing helps computers understand human language. 
    It combines linguistics and machine learning techniques.
    """
    
    chunks = chunker.chunk_text(sample_text, metadata={'source': 'example'})
    
    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Metadata: {chunk['metadata']}\n")

