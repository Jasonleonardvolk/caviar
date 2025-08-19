#!/usr/bin/env python3
"""
🔬 TEST: Natural Filtering vs Hard Cap
Let's see how many concepts we get WITHOUT the 50-cap
"""

import sys
from pathlib import Path

# Temporarily modify the cap
print("🔬 Testing natural consensus filtering...")
print("=" * 70)

# Read current pipeline.py
pipeline_path = Path("ingest_pdf/pipeline.py")
with open(pipeline_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Test 1: Current setup (with 50 cap)
print("\n📊 TEST 1: With 50-concept cap")
from ingest_pdf.pipeline import ingest_pdf_clean

# Create test document with mixed quality concepts
test_file = "test_consensus.txt"
with open(test_file, "w") as f:
    # High consensus terms (should pass)
    f.write("Quantum computing quantum computing quantum computing\n" * 5)
    f.write("Machine learning machine learning machine learning\n" * 5)
    f.write("Neural networks neural networks neural networks\n" * 5)
    
    # Medium quality (might pass)
    f.write("Artificial intelligence optimization algorithms\n" * 3)
    f.write("Deep learning backpropagation training\n" * 3)
    
    # Low quality (should be filtered)
    f.write("The document analysis method approach study\n" * 10)
    f.write("System model based using various different\n" * 10)

result1 = ingest_pdf_clean(test_file)

print(f"  Final concepts: {result1.get('concept_count', 0)}")
if result1.get('purity_analysis'):
    pa = result1['purity_analysis']
    print(f"  Raw → Pure: {pa.get('raw_concepts', 0)} → {pa.get('pure_concepts', 0)}")
    dist = pa.get('distribution', {})
    print(f"  Consensus: {dist.get('consensus', 0)}")
    print(f"  High conf: {dist.get('high_confidence', 0)}")
    print(f"  Single method: {dist.get('single_method', 0)}")

# Test 2: Without hard cap (modify temporarily)
print("\n📊 TEST 2: WITHOUT 50-concept cap (natural filtering only)")

# Backup and modify
backup_content = content
modified_content = content.replace(
    "MAX_USER_FRIENDLY_CONCEPTS = 50",
    "MAX_USER_FRIENDLY_CONCEPTS = 500  # Temporarily disabled"
)

# Write modified version
with open(pipeline_path, 'w', encoding='utf-8') as f:
    f.write(modified_content)

# Reload module
import importlib
import ingest_pdf.pipeline
importlib.reload(ingest_pdf.pipeline)
from ingest_pdf.pipeline import ingest_pdf_clean as ingest_pdf_clean_no_cap

result2 = ingest_pdf_clean_no_cap(test_file)

print(f"  Final concepts: {result2.get('concept_count', 0)}")
if result2.get('purity_analysis'):
    pa = result2['purity_analysis']
    print(f"  Raw → Pure: {pa.get('raw_concepts', 0)} → {pa.get('pure_concepts', 0)}")
    dist = pa.get('distribution', {})
    print(f"  Consensus: {dist.get('consensus', 0)}")
    print(f"  High conf: {dist.get('high_confidence', 0)}")
    print(f"  Single method: {dist.get('single_method', 0)}")

# Restore original
with open(pipeline_path, 'w', encoding='utf-8') as f:
    f.write(backup_content)

# Cleanup
Path(test_file).unlink()

print("\n" + "=" * 70)
print("🤔 ANALYSIS:")

if result1.get('concept_count', 0) == 50 and result2.get('concept_count', 0) > 50:
    print("❌ The 50-concept limit is ONLY from the hard cap!")
    print("   Natural filtering alone gives:", result2.get('concept_count', 0), "concepts")
    print("\n💡 RECOMMENDATION: Adjust consensus thresholds to be stricter")
elif result2.get('concept_count', 0) <= 60:
    print("✅ Natural filtering is working well!")
    print("   Only", result2.get('concept_count', 0), "concepts without hard cap")
    print("\n💡 The hard cap is just a safety net")
else:
    print("🤷 Mixed results - natural filtering helps but isn't enough alone")

print("\n📝 To make natural filtering stronger, consider:")
print("  1. Raise single-method threshold to 0.95+ (currently 0.9)")
print("  2. Require word_count = 1 for single-method (currently ≤2)")
print("  3. Increase consensus boost factors")
