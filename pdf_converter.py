"""
PDF to Text Converter
Standalone script to extract text from PDF files
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
import pdfplumber
from pypdf import PdfReader


class PDFConverter:
    """
    Convert PDF files to text with metadata extraction
    Uses pdfplumber for better text extraction and pypdf for metadata
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_from_pdf(
        self, 
        pdf_path: str,
        extract_images: bool = False
    ) -> Dict:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            extract_images: Whether to extract image information (default: False)
        
        Returns:
            Dictionary containing:
                - full_text: Complete text from all pages
                - pages: List of text per page
                - metadata: PDF metadata (author, title, etc.)
                - num_pages: Number of pages
                - file_info: File information
        """
        try:
            # Extract metadata using pypdf
            pdf_reader = PdfReader(pdf_path)
            pdf_metadata = pdf_reader.metadata if pdf_reader.metadata else {}
            
            metadata = {
                'title': pdf_metadata.get('/Title', ''),
                'author': pdf_metadata.get('/Author', ''),
                'subject': pdf_metadata.get('/Subject', ''),
                'keywords': pdf_metadata.get('/Keywords', ''),
                'creator': pdf_metadata.get('/Creator', ''),
                'producer': pdf_metadata.get('/Producer', ''),
                'creation_date': str(pdf_metadata.get('/CreationDate', '')),
                'mod_date': str(pdf_metadata.get('/ModDate', '')),
            }
            
            # Extract text using pdfplumber (better text extraction)
            pages = []
            full_text = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    
                    pages.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'char_count': len(text),
                        'word_count': len(text.split())
                    })
                    
                    full_text.append(text)
            
            # Get file information
            file_path = Path(pdf_path)
            file_info = {
                'filename': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_path': str(file_path.absolute())
            }
            
            # Combine all text
            complete_text = "\n\n".join(full_text)
            
            return {
                'full_text': complete_text,
                'pages': pages,
                'metadata': metadata,
                'num_pages': len(pages),
                'file_info': file_info,
                'total_chars': len(complete_text),
                'total_words': len(complete_text.split())
            }
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def save_text_to_file(
        self, 
        text: str, 
        output_path: str
    ) -> str:
        """
        Save extracted text to a file
        
        Args:
            text: Text content to save
            output_path: Path where to save the text file
        
        Returns:
            Path to the saved file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            return str(output_file.absolute())
            
        except Exception as e:
            raise Exception(f"Error saving text to file: {str(e)}")
    
    def convert_pdf_to_txt(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
        save_pages_separately: bool = False
    ) -> Dict:
        """
        Convert PDF to text file(s)
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output text file (optional, auto-generated if not provided)
            save_pages_separately: Save each page as a separate file
        
        Returns:
            Dictionary with conversion results
        """
        # Extract text
        result = self.extract_text_from_pdf(pdf_path)
        
        # Generate output path if not provided
        if output_path is None:
            pdf_file = Path(pdf_path)
            output_path = pdf_file.with_suffix('.txt')
        
        # Save full text
        saved_file = self.save_text_to_file(result['full_text'], output_path)
        
        result['output_file'] = saved_file
        
        # Optionally save pages separately
        if save_pages_separately:
            output_dir = Path(output_path).parent / f"{Path(pdf_path).stem}_pages"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            page_files = []
            for page_data in result['pages']:
                page_num = page_data['page_number']
                page_file = output_dir / f"page_{page_num:04d}.txt"
                self.save_text_to_file(page_data['text'], str(page_file))
                page_files.append(str(page_file))
            
            result['page_files'] = page_files
        
        return result
    
    def batch_convert(
        self,
        pdf_directory: str,
        output_directory: Optional[str] = None,
        recursive: bool = False
    ) -> List[Dict]:
        """
        Batch convert multiple PDFs in a directory
        
        Args:
            pdf_directory: Directory containing PDF files
            output_directory: Directory for output text files (optional)
            recursive: Search subdirectories recursively
        
        Returns:
            List of conversion results
        """
        pdf_dir = Path(pdf_directory)
        
        if not pdf_dir.exists():
            raise ValueError(f"Directory not found: {pdf_directory}")
        
        # Find all PDF files
        if recursive:
            pdf_files = list(pdf_dir.rglob("*.pdf"))
        else:
            pdf_files = list(pdf_dir.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files")
        
        results = []
        for pdf_file in pdf_files:
            try:
                print(f"\nProcessing: {pdf_file.name}")
                
                # Generate output path
                if output_directory:
                    output_dir = Path(output_directory)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / pdf_file.with_suffix('.txt').name
                else:
                    output_path = pdf_file.with_suffix('.txt')
                
                # Convert
                result = self.convert_pdf_to_txt(str(pdf_file), str(output_path))
                results.append({
                    'status': 'success',
                    'input_file': str(pdf_file),
                    'output_file': result['output_file'],
                    'num_pages': result['num_pages'],
                    'total_words': result['total_words']
                })
                
                print(f"✓ Converted: {result['num_pages']} pages, {result['total_words']} words")
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                results.append({
                    'status': 'error',
                    'input_file': str(pdf_file),
                    'error': str(e)
                })
        
        return results


def main():
    """
    Command-line interface for PDF conversion
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert PDF files to text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single PDF
  python pdf_converter.py input.pdf
  
  # Convert with custom output path
  python pdf_converter.py input.pdf -o output.txt
  
  # Save pages separately
  python pdf_converter.py input.pdf --pages
  
  # Batch convert directory
  python pdf_converter.py --batch ./pdfs --output ./texts
  
  # Batch convert recursively
  python pdf_converter.py --batch ./pdfs --output ./texts --recursive
        """
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Input PDF file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output text file path (optional)'
    )
    
    parser.add_argument(
        '--pages',
        action='store_true',
        help='Save each page as a separate file'
    )
    
    parser.add_argument(
        '--batch',
        help='Batch convert directory of PDFs'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search subdirectories recursively (for batch mode)'
    )
    
    args = parser.parse_args()
    
    converter = PDFConverter()
    
    # Batch mode
    if args.batch:
        print(f"Batch converting PDFs from: {args.batch}")
        results = converter.batch_convert(
            args.batch,
            args.output,
            args.recursive
        )
        
        # Summary
        success_count = sum(1 for r in results if r['status'] == 'success')
        print(f"\n{'='*50}")
        print(f"Summary: {success_count}/{len(results)} files converted successfully")
        
        return
    
    # Single file mode
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    try:
        print(f"Converting: {args.input}")
        
        result = converter.convert_pdf_to_txt(
            args.input,
            args.output,
            args.pages
        )
        
        print(f"\n{'='*50}")
        print("Conversion successful!")
        print(f"{'='*50}")
        print(f"Input file:    {result['file_info']['filename']}")
        print(f"Output file:   {result['output_file']}")
        print(f"Pages:         {result['num_pages']}")
        print(f"Total words:   {result['total_words']:,}")
        print(f"Total chars:   {result['total_chars']:,}")
        
        if args.pages:
            print(f"Page files:    {len(result['page_files'])} files created")
        
        # Display metadata if available
        if result['metadata']['title']:
            print(f"\nMetadata:")
            print(f"  Title:  {result['metadata']['title']}")
            if result['metadata']['author']:
                print(f"  Author: {result['metadata']['author']}")
        
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

