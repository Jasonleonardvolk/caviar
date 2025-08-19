#!/usr/bin/env python3
"""
Practical examples of using TORI's dynamic configuration system.
This shows real-world scenarios for different deployment environments.
"""

import os
from pathlib import Path
from ingest_pdf.pipeline.config import settings, Settings
from ingest_pdf.pipeline import ingest_pdf_clean

def example_development_setup():
    """Example: Development environment with verbose logging and small limits"""
    print("=== Development Environment Setup ===")
    
    # Option 1: Using environment variables
    os.environ.update({
        'MAX_PARALLEL_WORKERS': '4',  # Limit for local development
        'ENTROPY_THRESHOLD': '0.0001',  # Less aggressive pruning for testing
        'SMALL_FILE_MB': '0.5',  # Lower limits for quick testing
        'ENABLE_EMOJI_LOGS': 'true',  # Fun logs for development
    })
    
    # Create new settings instance
    dev_settings = Settings()
    print(f"Dev max workers: {dev_settings.max_parallel_workers}")
    print(f"Dev entropy threshold: {dev_settings.entropy_threshold}")
    
    # Process a test file
    # result = ingest_pdf_clean("test_document.pdf")


def example_production_setup():
    """Example: Production environment with optimized settings"""
    print("\n=== Production Environment Setup ===")
    
    # Production settings via environment
    prod_env = {
        'MAX_PARALLEL_WORKERS': '32',  # Use more cores in production
        'ENTROPY_THRESHOLD': '0.00005',  # More aggressive pruning
        'ENABLE_EMOJI_LOGS': 'false',  # Clean logs for log aggregators
        'OCR_MAX_PAGES': '100',  # Limit OCR to prevent runaway processing
        'LARGE_FILE_MB': '50',  # Handle larger files in production
        'LARGE_CONCEPTS': '3000',  # More concepts for large files
    }
    
    # Simulate production environment
    for key, value in prod_env.items():
        os.environ[key] = value
    
    prod_settings = Settings()
    print(f"Prod max workers: {prod_settings.max_parallel_workers}")
    print(f"Prod entropy threshold: {prod_settings.entropy_threshold}")
    print(f"Prod OCR limit: {prod_settings.ocr_max_pages} pages")


def example_memory_constrained_setup():
    """Example: Memory-constrained environment (e.g., container with limited RAM)"""
    print("\n=== Memory-Constrained Environment ===")
    
    # Conservative settings for limited memory
    os.environ.update({
        'MAX_PARALLEL_WORKERS': '2',  # Minimize memory usage
        'SMALL_CHUNKS': '200',  # Smaller chunks
        'MEDIUM_CHUNKS': '400',
        'LARGE_CHUNKS': '800',
        'MAX_DIVERSE_CONCEPTS': '1000',  # Cap total concepts
        'ENABLE_PARALLEL_PROCESSING': 'false',  # Sequential processing
    })
    
    mem_settings = Settings()
    print(f"Memory-safe workers: {mem_settings.max_parallel_workers}")
    print(f"Large file chunks: {mem_settings.large_chunks}")
    print(f"Parallel processing: {mem_settings.enable_parallel_processing}")


def example_quality_focused_setup():
    """Example: Quality-focused setup for research papers"""
    print("\n=== Quality-Focused Setup ===")
    
    # Emphasize quality over speed
    quality_config = {
        'ENTROPY_THRESHOLD': '0.0002',  # Keep more diverse concepts
        'SIMILARITY_THRESHOLD': '0.95',  # Only remove very similar concepts
        'SECTION_WEIGHTS_JSON': json.dumps({
            "title": 3.0,  # Heavily weight title
            "abstract": 2.5,  # And abstract
            "methodology": 2.0,  # Important for research
            "conclusion": 2.0,
            "introduction": 1.5,
            "results": 1.5,
            "discussion": 1.2,
            "body": 1.0,
            "references": 0.3  # De-emphasize references
        }),
        'ENABLE_CONTEXT_EXTRACTION': 'true',
        'ENABLE_SMART_FILTERING': 'true',
        'ENABLE_ENHANCED_MEMORY_STORAGE': 'true',
    }
    
    for key, value in quality_config.items():
        os.environ[key] = value
    
    quality_settings = Settings()
    print("Section weights for quality focus:")
    for section, weight in quality_settings.section_weights.items():
        print(f"  {section}: {weight}")


def example_speed_optimized_setup():
    """Example: Speed-optimized setup for bulk processing"""
    print("\n=== Speed-Optimized Setup ===")
    
    # Optimize for throughput
    speed_config = {
        'MAX_PARALLEL_WORKERS': '64',  # Max parallelism
        'ENABLE_OCR_FALLBACK': 'false',  # Skip OCR for speed
        'ENABLE_ENTROPY_PRUNING': 'false',  # Skip pruning
        'ENABLE_SMART_FILTERING': 'false',  # Minimal filtering
        'SMALL_CONCEPTS': '100',  # Fewer concepts
        'MEDIUM_CONCEPTS': '300',
        'LARGE_CONCEPTS': '500',
    }
    
    for key, value in speed_config.items():
        os.environ[key] = value
    
    speed_settings = Settings()
    print(f"Speed workers: {speed_settings.max_parallel_workers}")
    print(f"OCR enabled: {speed_settings.enable_ocr_fallback}")
    print(f"Entropy pruning: {speed_settings.enable_entropy_pruning}")


def example_docker_compose():
    """Example: Docker Compose configuration"""
    print("\n=== Docker Compose Example ===")
    print("""
version: '3.8'
services:
  tori-pipeline:
    image: tori/pipeline:latest
    environment:
      # Production optimized settings
      - MAX_PARALLEL_WORKERS=16
      - ENTROPY_THRESHOLD=0.00005
      - SIMILARITY_THRESHOLD=0.88
      - ENABLE_EMOJI_LOGS=false
      - OCR_MAX_PAGES=50
      # File size limits
      - LARGE_FILE_MB=100
      - LARGE_CHUNKS=3000
      - LARGE_CONCEPTS=5000
      # Section weights as JSON
      - SECTION_WEIGHTS_JSON={"title":2.5,"abstract":2.0,"methodology":1.5}
    volumes:
      - ./pdfs:/app/pdfs
      - ./output:/app/output
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
""")


def example_kubernetes_configmap():
    """Example: Kubernetes ConfigMap"""
    print("\n=== Kubernetes ConfigMap Example ===")
    print("""
apiVersion: v1
kind: ConfigMap
metadata:
  name: tori-pipeline-config
data:
  # Feature flags
  ENABLE_CONTEXT_EXTRACTION: "true"
  ENABLE_ENTROPY_PRUNING: "true"
  ENABLE_OCR_FALLBACK: "true"
  ENABLE_PARALLEL_PROCESSING: "true"
  
  # Performance tuning
  MAX_PARALLEL_WORKERS: "32"
  ENTROPY_THRESHOLD: "0.00005"
  
  # File limits for cloud environment
  LARGE_FILE_MB: "200"
  XLARGE_CHUNKS: "5000"
  XLARGE_CONCEPTS: "10000"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tori-pipeline
spec:
  template:
    spec:
      containers:
      - name: pipeline
        image: tori/pipeline:latest
        envFrom:
        - configMapRef:
            name: tori-pipeline-config
""")


def example_dynamic_tuning():
    """Example: Dynamic tuning based on file characteristics"""
    print("\n=== Dynamic Tuning Example ===")
    
    from ingest_pdf.pipeline.config import settings, Settings
    
    def process_pdf_with_dynamic_config(pdf_path: Path):
        """Adjust configuration based on file characteristics"""
        
        # Get file size
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        
        # Create custom settings based on file size
        if file_size_mb < 1:
            # Small file - prioritize quality
            os.environ['ENTROPY_THRESHOLD'] = '0.0002'
            os.environ['MAX_PARALLEL_WORKERS'] = '4'
        elif file_size_mb < 10:
            # Medium file - balanced
            os.environ['ENTROPY_THRESHOLD'] = '0.0001'
            os.environ['MAX_PARALLEL_WORKERS'] = '8'
        else:
            # Large file - prioritize speed
            os.environ['ENTROPY_THRESHOLD'] = '0.00005'
            os.environ['MAX_PARALLEL_WORKERS'] = '16'
            os.environ['ENABLE_OCR_FALLBACK'] = 'false'  # Skip OCR for large files
        
        # Process with custom settings
        custom_settings = Settings()
        print(f"Processing {pdf_path.name} ({file_size_mb:.1f}MB)")
        print(f"  - Workers: {custom_settings.max_parallel_workers}")
        print(f"  - Entropy threshold: {custom_settings.entropy_threshold}")
        
        # result = ingest_pdf_clean(str(pdf_path))
        # return result
    
    # Example usage
    example_path = Path("example.pdf")
    # process_pdf_with_dynamic_config(example_path)


import json

if __name__ == "__main__":
    # Clean environment first
    env_vars_to_clean = [
        'MAX_PARALLEL_WORKERS', 'ENTROPY_THRESHOLD', 'SIMILARITY_THRESHOLD',
        'ENABLE_EMOJI_LOGS', 'OCR_MAX_PAGES', 'SMALL_FILE_MB',
        'LARGE_FILE_MB', 'LARGE_CONCEPTS', 'SMALL_CHUNKS', 'MEDIUM_CHUNKS',
        'LARGE_CHUNKS', 'MAX_DIVERSE_CONCEPTS', 'ENABLE_PARALLEL_PROCESSING',
        'SECTION_WEIGHTS_JSON', 'ENABLE_CONTEXT_EXTRACTION', 'ENABLE_SMART_FILTERING',
        'ENABLE_ENHANCED_MEMORY_STORAGE', 'ENABLE_OCR_FALLBACK', 'ENABLE_ENTROPY_PRUNING',
        'SMALL_CONCEPTS', 'MEDIUM_CONCEPTS', 'XLARGE_CHUNKS', 'XLARGE_CONCEPTS'
    ]
    for var in env_vars_to_clean:
        os.environ.pop(var, None)
    
    # Run examples
    example_development_setup()
    example_production_setup()
    example_memory_constrained_setup()
    example_quality_focused_setup()
    example_speed_optimized_setup()
    example_docker_compose()
    example_kubernetes_configmap()
    example_dynamic_tuning()
    
    print("\n✅ All configuration examples completed!")
