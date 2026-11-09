"""
Link Processing Module
Extracts URLs from documents, fetches their content, and enriches the indexed data
"""

import re
import requests
from typing import List, Dict, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import trafilatura
import tempfile
from pathlib import Path
from pdf_converter import PDFConverter


class LinkProcessor:
    """
    Process links found in documents:
    - Extract URLs (websites, PDFs, documents)
    - Fetch content from web pages
    - Download and extract PDFs
    - Return enriched content for indexing
    """
    
    def __init__(
        self,
        timeout: int = 10,
        max_retries: int = 2,
        user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        download_websites: bool = False,  # Disabled by default (Gemini handles grounding)
        download_pdfs: bool = True  # Keep PDFs enabled by default
    ):
        """
        Initialize link processor
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            user_agent: User agent for requests (defaults to recent Chrome)
            download_websites: Whether to download HTML website content (default: False)
            download_pdfs: Whether to download PDF content (default: True)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.download_websites = download_websites
        self.download_pdfs = download_pdfs
        self.headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,tr;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.pdf_converter = PDFConverter()
    
    def extract_urls(self, text: str) -> List[Dict[str, str]]:
        """
        Extract all URLs from text
        
        Args:
            text: Text containing URLs
        
        Returns:
            List of dicts with url and type (website/pdf)
        """
        # First, normalize the text by removing line breaks within URLs
        # This handles cases where URLs are split across multiple lines
        # Keep the hyphen if it's followed by a newline and more URL content
        normalized_text = re.sub(r'-\s*\n\s*(?=[a-zA-Z0-9])', '-', text)
        # Also handle chrome-extension URLs split across lines
        normalized_text = re.sub(r'chrome-\s*\n\s*extension://', r'chrome-extension://', normalized_text)
        
        # Pattern to match URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        
        urls = []
        seen_urls = set()  # Track seen URLs to avoid duplicates
        
        for match in re.finditer(url_pattern, normalized_text):
            url = match.group(0)
            
            # Clean up URL (remove trailing punctuation)
            url = url.rstrip('.,;:!?)')
            
            # Handle Chrome extension PDF URLs - extract the actual PDF URL
            if 'efaidnbmnnnibpcajpcglclefindmkaj' in url:
                # Extract actual PDF URL from chrome extension format
                pdf_match = re.search(r'efaidnbmnnnibpcajpcglclefindmkaj/(https?://[^\s<>"{}|\\^`\[\]]+)', url)
                if pdf_match:
                    url = pdf_match.group(1)
            
            # Skip if we've already seen this URL
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            # Determine type
            url_lower = url.lower()
            if url_lower.endswith('.pdf') or '/pdf' in url_lower:
                url_type = 'pdf'
            else:
                url_type = 'website'
            
            urls.append({
                'url': url,
                'type': url_type,
                'original_text': match.group(0)
            })
        
        return urls
    
    def fetch_website_content(self, url: str) -> Optional[Dict[str, str]]:
        """
        Fetch and extract text content from a website
        
        Args:
            url: Website URL
        
        Returns:
            Dict with title, text, and metadata
        """
        try:
            print(f"  → Downloading website: {url}")
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            print(f"  → Response status: {response.status_code}, Content-Type: {response.headers.get('content-type', 'unknown')}")
            
            # Use trafilatura for intelligent content extraction
            # It automatically finds main content and removes boilerplate, ads, navigation
            downloaded = trafilatura.extract(
                response.text,
                output_format='txt',
                include_comments=False,
                include_tables=True,
                include_links=False,
                no_fallback=False  # Use fallback methods if needed
            )
            
            # Get title using BeautifulSoup as fallback
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ''
            
            # If trafilatura extraction failed, fall back to BeautifulSoup
            if not downloaded or len(downloaded.strip()) < 50:
                print(f"  → Trafilatura extraction minimal, using BeautifulSoup fallback")
                for element in soup(['script', 'style', 'noscript', 'iframe']):
                    element.decompose()
                body = soup.find('body')
                if body:
                    text = body.get_text(separator='\n', strip=True)
                else:
                    text = soup.get_text(separator='\n', strip=True)
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = '\n'.join(lines)
            else:
                text = downloaded.strip()
            
            print(f"  → Extracted {len(text)} characters from website")
            
            return {
                'title': title.strip() if title else 'No title',
                'text': text,
                'url': url,
                'status': 'success',
                'content_type': 'website',
                'char_count': len(text),
                'was_truncated': False
            }
        
        except Exception as e:
            print(f"  ✗ Error fetching website {url}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'url': url,
                'status': 'failed',
                'error': str(e),
                'content_type': 'website'
            }
    
    def fetch_pdf_content(self, url: str) -> Optional[Dict[str, str]]:
        """
        Download PDF and extract text content
        
        Args:
            url: PDF URL
        
        Returns:
            Dict with text and metadata
        """
        try:
            print(f"  → Downloading PDF: {url}")
            # Download PDF to temporary file
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout * 2,  # PDFs can be larger
                allow_redirects=True
            )
            response.raise_for_status()
            
            print(f"  → PDF downloaded: {len(response.content)} bytes")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            try:
                # Extract text from PDF
                print(f"  → Extracting text from PDF...")
                result = self.pdf_converter.extract_text_from_pdf(tmp_path)
                
                # Get full text - no truncation
                text = result['full_text']
                
                print(f"  → Extracted {len(text)} characters from PDF ({result['num_pages']} pages)")
                
                return {
                    'title': result['metadata'].get('title', 'PDF Document'),
                    'text': text,
                    'url': url,
                    'num_pages': result['num_pages'],
                    'status': 'success',
                    'content_type': 'pdf',
                    'char_count': len(text),
                    'was_truncated': False
                }
            
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
        
        except Exception as e:
            print(f"  ✗ Error fetching PDF {url}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'url': url,
                'status': 'failed',
                'error': str(e),
                'content_type': 'pdf'
            }
    
    def process_document_with_links(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Process a document: extract links, fetch their content, and return enriched data
        
        Args:
            text: Original document text
            metadata: Original metadata
        
        Returns:
            Dict with original text, linked content, and combined text
        """
        # Extract URLs
        urls = self.extract_urls(text)
        
        if not urls:
            return {
                'original_text': text,
                'linked_content': [],
                'enriched_text': text,
                'urls_found': 0,
                'urls_fetched': 0
            }
        
        print(f"Found {len(urls)} URLs in document")
        
        # Fetch content from URLs (based on configuration)
        linked_content = []
        for url_info in urls:
            # Skip based on download settings
            if url_info['type'] == 'pdf' and not self.download_pdfs:
                print(f"Skipping PDF (download_pdfs=False): {url_info['url']}")
                continue
            elif url_info['type'] == 'website' and not self.download_websites:
                print(f"Skipping website (download_websites=False): {url_info['url']}")
                continue
            
            print(f"Fetching {url_info['type']}: {url_info['url']}")
            
            if url_info['type'] == 'pdf':
                content = self.fetch_pdf_content(url_info['url'])
            else:
                content = self.fetch_website_content(url_info['url'])
            
            if content and content.get('status') == 'success':
                linked_content.append(content)
        
        # Create enriched text combining original + linked content
        enriched_parts = [text]
        
        for i, content in enumerate(linked_content):
            if content.get('status') == 'success':
                enriched_parts.append(
                    f"\n\n--- Linked Content {i+1}: {content['url']} ---\n"
                    f"Title: {content.get('title', 'N/A')}\n"
                    f"Content:\n{content['text']}\n"
                    f"--- End of Linked Content {i+1} ---\n"
                )
        
        enriched_text = '\n'.join(enriched_parts)
        
        return {
            'original_text': text,
            'linked_content': linked_content,
            'enriched_text': enriched_text,
            'urls_found': len(urls),
            'urls_fetched': len(linked_content),
            'url_details': urls
        }
    
    def process_chunk_with_links(
        self,
        chunk_text: str,
        chunk_metadata: Dict
    ) -> Dict[str, any]:
        """
        Process a single chunk: extract and fetch links if present
        
        Args:
            chunk_text: Chunk text
            chunk_metadata: Chunk metadata
        
        Returns:
            Dict with enriched chunk data
        """
        urls = self.extract_urls(chunk_text)
        
        if not urls:
            return {
                'text': chunk_text,
                'metadata': chunk_metadata,
                'has_links': False
            }
        
        # Process links
        result = self.process_document_with_links(chunk_text, chunk_metadata)
        
        # Update metadata
        enriched_metadata = chunk_metadata.copy()
        enriched_metadata['has_links'] = True
        enriched_metadata['urls_found'] = result['urls_found']
        enriched_metadata['urls_fetched'] = result['urls_fetched']
        
        return {
            'text': result['enriched_text'],
            'metadata': enriched_metadata,
            'has_links': True,
            'linked_content': result['linked_content']
        }


# Example usage
if __name__ == "__main__":
    processor = LinkProcessor()
    
    # Example text with links
    sample_text = """
    IPO
    Date: 09.09.2025
    Subject: DOF Robotics IPO
    Link1: https://dofrobotics.com/tr/hakkimizda/
    DOF Robotics is a high-tech company.
    Link2: https://example.com/document.pdf
    """
    
    result = processor.process_document_with_links(sample_text)
    
    print(f"URLs found: {result['urls_found']}")
    print(f"URLs fetched: {result['urls_fetched']}")
    print(f"\nEnriched text length: {len(result['enriched_text'])} chars")
    print(f"\nFirst 500 chars of enriched text:")
    print(result['enriched_text'][:500])

