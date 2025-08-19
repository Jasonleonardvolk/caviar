"""
Example: Using NLTK for enhanced text processing in TORI ingestion pipeline.

This module shows how to integrate NLTK sentence-level features
into the existing PDF ingestion process.
"""

import nltk
from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class NLTKTextProcessor:
    """Enhanced text processor using NLTK for sentence-level analysis."""
    
    def __init__(self):
        """Initialize NLTK processor and download data if needed."""
        try:
            # Check if data is available
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Cache stopwords for efficiency
        self.stop_words = set(stopwords.words('english'))
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text with NLTK features.
        
        Args:
            text: Raw text to process
            
        Returns:
            Dictionary with processed results
        """
        # Sentence tokenization
        sentences = sent_tokenize(text)
        
        # Word tokenization and filtering
        words = word_tokenize(text.lower())
        filtered_words = [w for w in words if w.isalnum() and w not in self.stop_words]
        
        # Extract key sentences (e.g., first and last)
        key_sentences = []
        if sentences:
            key_sentences.append(sentences[0])  # First sentence
            if len(sentences) > 1:
                key_sentences.append(sentences[-1])  # Last sentence
        
        # Calculate statistics
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences) if sentences else 0
        
        return {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'unique_words': len(set(filtered_words)),
            'avg_sentence_length': avg_sentence_length,
            'key_sentences': key_sentences,
            'sentences': sentences,  # All sentences for further processing
            'filtered_words': filtered_words[:100]  # Top 100 non-stop words
        }
    
    def extract_chunks(self, text: str, chunk_size: int = 5) -> List[str]:
        """
        Extract text chunks based on sentence boundaries.
        
        Args:
            text: Text to chunk
            chunk_size: Number of sentences per chunk
            
        Returns:
            List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def get_sentence_embeddings_input(self, text: str, max_sentences: int = 50) -> List[str]:
        """
        Prepare sentences for embedding generation.
        
        Args:
            text: Source text
            max_sentences: Maximum number of sentences to return
            
        Returns:
            List of sentences ready for embedding
        """
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences
        meaningful_sentences = [
            sent for sent in sentences 
            if len(sent.split()) >= 5  # At least 5 words
        ]
        
        return meaningful_sentences[:max_sentences]


# Integration example for the existing ingestion pipeline
def enhance_pdf_ingestion_with_nltk(extracted_text: str) -> Dict[str, Any]:
    """
    Example of integrating NLTK processing into PDF ingestion.
    
    This would be called from the existing PDF extraction pipeline
    to add sentence-level features.
    """
    processor = NLTKTextProcessor()
    
    # Process the text
    results = processor.process_text(extracted_text)
    
    # Create sentence-based chunks for better context preservation
    chunks = processor.extract_chunks(extracted_text, chunk_size=3)
    
    # Get sentences for embedding
    embedding_sentences = processor.get_sentence_embeddings_input(extracted_text)
    
    return {
        'statistics': results,
        'chunks': chunks,
        'embedding_sentences': embedding_sentences,
        'metadata': {
            'processing_method': 'nltk_enhanced',
            'sentence_count': results['sentence_count'],
            'avg_sentence_length': results['avg_sentence_length']
        }
    }


if __name__ == "__main__":
    # Test the processor
    test_text = """
    TORI is a consciousness interface system. It processes documents and extracts meaningful concepts.
    The system uses advanced natural language processing. This includes sentence tokenization and analysis.
    By understanding text at the sentence level, TORI can better preserve context and meaning.
    """
    
    processor = NLTKTextProcessor()
    results = processor.process_text(test_text)
    
    print("=== NLTK Text Processing Results ===")
    print(f"Sentences found: {results['sentence_count']}")
    print(f"Average sentence length: {results['avg_sentence_length']:.1f} words")
    print(f"Unique non-stop words: {results['unique_words']}")
    print("\nKey sentences:")
    for i, sent in enumerate(results['key_sentences'], 1):
        print(f"  {i}. {sent}")
    
    print("\n=== Sentence-based Chunks ===")
    chunks = processor.extract_chunks(test_text, chunk_size=2)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
