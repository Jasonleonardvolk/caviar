#!/usr/bin/env python3
"""
Quick start script for TORI Dynamic Configuration
Run this to see the configuration system in action!
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print('=' * 60)

def main():
    print_section("TORI DYNAMIC CONFIGURATION DEMO")
    
    # 1. Show default configuration
    print("\n1️⃣ DEFAULT CONFIGURATION")
    print("-" * 40)
    
    from ingest_pdf.pipeline.config import settings
    print(f"Entropy Pruning: {settings.enable_entropy_pruning}")
    print(f"Max Workers: {settings.max_parallel_workers or 'auto-detect'}")
    print(f"Entropy Threshold: {settings.entropy_threshold}")
    print(f"OCR Max Pages: {settings.ocr_max_pages or 'unlimited'}")
    
    # 2. Override with environment variable
    print("\n2️⃣ OVERRIDE WITH ENVIRONMENT VARIABLE")
    print("-" * 40)
    
    print("Setting MAX_PARALLEL_WORKERS=8...")
    os.environ['MAX_PARALLEL_WORKERS'] = '8'
    
    # Need to reimport to get new settings
    from ingest_pdf.pipeline.config import Settings
    new_settings = Settings()
    print(f"Max Workers is now: {new_settings.max_parallel_workers}")
    
    # 3. Show backward compatibility
    print("\n3️⃣ BACKWARD COMPATIBILITY")
    print("-" * 40)
    
    from ingest_pdf.pipeline.config import MAX_PARALLEL_WORKERS, ENABLE_ENTROPY_PRUNING
    print(f"Old-style MAX_PARALLEL_WORKERS: {MAX_PARALLEL_WORKERS}")
    print(f"Old-style ENABLE_ENTROPY_PRUNING: {ENABLE_ENTROPY_PRUNING}")
    print("✅ Existing code continues to work!")
    
    # 4. Show complex configuration
    print("\n4️⃣ COMPLEX CONFIGURATION (Section Weights)")
    print("-" * 40)
    
    # Set section weights as JSON
    os.environ['SECTION_WEIGHTS_JSON'] = '{"title": 3.0, "abstract": 2.5, "introduction": 1.5}'
    test_settings = Settings()
    
    print("Section weights after JSON override:")
    for section, weight in sorted(test_settings.section_weights.items()):
        print(f"  {section:15} = {weight}")
    
    # 5. Show file size limits
    print("\n5️⃣ FILE SIZE LIMITS")
    print("-" * 40)
    
    from ingest_pdf.pipeline.config import FILE_SIZE_LIMITS
    for size_cat, (max_bytes, chunks, concepts) in FILE_SIZE_LIMITS.items():
        if max_bytes == float('inf'):
            print(f"{size_cat:8} > {FILE_SIZE_LIMITS['large'][0]/(1024*1024):.0f}MB: "
                  f"{chunks:4} chunks, {concepts:4} concepts")
        else:
            print(f"{size_cat:8} < {max_bytes/(1024*1024):.0f}MB: "
                  f"{chunks:4} chunks, {concepts:4} concepts")
    
    # 6. Show configuration for different environments
    print("\n6️⃣ ENVIRONMENT-SPECIFIC CONFIGURATIONS")
    print("-" * 40)
    
    # Clean environment
    for key in ['MAX_PARALLEL_WORKERS', 'ENTROPY_THRESHOLD', 'ENABLE_OCR_FALLBACK']:
        os.environ.pop(key, None)
    
    # Development config
    print("\n📝 Development Configuration:")
    dev_env = {
        'MAX_PARALLEL_WORKERS': '4',
        'ENTROPY_THRESHOLD': '0.0001',
        'ENABLE_OCR_FALLBACK': 'true'
    }
    for k, v in dev_env.items():
        os.environ[k] = v
    dev_settings = Settings()
    print(f"  Workers: {dev_settings.max_parallel_workers}, "
          f"Entropy: {dev_settings.entropy_threshold}, "
          f"OCR: {dev_settings.enable_ocr_fallback}")
    
    # Production config
    print("\n🚀 Production Configuration:")
    prod_env = {
        'MAX_PARALLEL_WORKERS': '32',
        'ENTROPY_THRESHOLD': '0.00005',
        'ENABLE_OCR_FALLBACK': 'false'
    }
    for k, v in prod_env.items():
        os.environ[k] = v
    prod_settings = Settings()
    print(f"  Workers: {prod_settings.max_parallel_workers}, "
          f"Entropy: {prod_settings.entropy_threshold}, "
          f"OCR: {prod_settings.enable_ocr_fallback}")
    
    # 7. Create example .env file
    print("\n7️⃣ CREATING EXAMPLE .env FILE")
    print("-" * 40)
    
    env_content = """# Example .env file for TORI Pipeline
# Copy to .env and modify as needed

# Performance
MAX_PARALLEL_WORKERS=16
ENTROPY_THRESHOLD=0.00008

# Features
ENABLE_OCR_FALLBACK=true
ENABLE_ENTROPY_PRUNING=true

# File limits
LARGE_FILE_MB=50
LARGE_CONCEPTS=3000
"""
    
    env_path = Path(__file__).parent / '.env.demo'
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"Created {env_path}")
    print("You can copy this to .env and modify for your needs")
    
    # Summary
    print_section("✅ CONFIGURATION SYSTEM READY!")
    
    print("""
Next steps:
1. Copy .env.example to .env and customize
2. Set environment variables in your deployment
3. Use settings object in new code
4. Existing code continues to work unchanged

For more info:
- Read DYNAMIC_CONFIG_README.md
- Check dynamic_config_examples.py
- See MIGRATION_GUIDE.md
""")

if __name__ == "__main__":
    main()
