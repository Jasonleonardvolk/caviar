#!/usr/bin/env python3
"""
🔬 TEST: Natural Filtering vs Hard Cap - Using Real PDF
"""

import os
from pathlib import Path

print("🔬 Testing natural consensus filtering with real PDF...")
print("=" * 70)

# Find a real PDF to test with
pdf_files = list(Path(".").glob("*.pdf"))
if not pdf_files:
    pdf_files = list(Path("test_pdfs").glob("*.pdf"))
if not pdf_files:
    print("❌ No PDF files found to test with!")
    print("Please run this test with a real PDF file:")
    print("  python test_with_real_pdf.py your_file.pdf")
    exit(1)

test_pdf = pdf_files[0]
print(f"📄 Using test PDF: {test_pdf}")

# Test 1: Current setup (with 50 cap)
print("\n📊 TEST 1: With 50-concept cap")
from ingest_pdf.pipeline import ingest_pdf_clean

result1 = ingest_pdf_clean(str(test_pdf))

print(f"  Final concepts: {result1.get('concept_count', 0)}")
if result1.get('purity_analysis'):
    pa = result1['purity_analysis']
    print(f"  Raw → Pure: {pa.get('raw_concepts', 0)} → {pa.get('pure_concepts', 0)}")
    dist = pa.get('distribution', {})
    print(f"  Consensus: {dist.get('consensus', 0)}")
    print(f"  High conf: {dist.get('high_confidence', 0)}")
    print(f"  Database boosted: {dist.get('file_storage_boosted', 0)}")
    print(f"  Single method: {dist.get('single_method', 0)}")

# Test 2: Without hard cap (modify temporarily)
print("\n📊 TEST 2: WITHOUT 50-concept cap (natural filtering only)")

# Read and modify pipeline
pipeline_path = Path("ingest_pdf/pipeline.py")
with open(pipeline_path, 'r', encoding='utf-8') as f:
    original_content = f.read()

modified_content = original_content.replace(
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

result2 = ingest_pdf_clean_no_cap(str(test_pdf))

print(f"  Final concepts: {result2.get('concept_count', 0)}")
if result2.get('purity_analysis'):
    pa = result2['purity_analysis']
    print(f"  Raw → Pure: {pa.get('raw_concepts', 0)} → {pa.get('pure_concepts', 0)}")
    dist = pa.get('distribution', {})
    print(f"  Consensus: {dist.get('consensus', 0)}")
    print(f"  High conf: {dist.get('high_confidence', 0)}")
    print(f"  Database boosted: {dist.get('file_storage_boosted', 0)}")
    print(f"  Single method: {dist.get('single_method', 0)}")

# Restore original
with open(pipeline_path, 'w', encoding='utf-8') as f:
    f.write(original_content)

print("\n" + "=" * 70)
print("🤔 ANALYSIS:")

# Calculate the difference
with_cap = result1.get('concept_count', 0)
without_cap = result2.get('concept_count', 0)

if with_cap == 50 and without_cap > 50:
    excess = without_cap - 50
    print(f"❌ The 50-concept limit is cutting off {excess} concepts!")
    print(f"   Natural filtering gives: {without_cap} concepts")
    print(f"   We're forcing it down to: {with_cap} concepts")
    
    if without_cap > 100:
        print("\n⚠️ Natural filtering is NOT strict enough!")
        print("\n💡 RECOMMENDATIONS:")
        print("  1. Increase single-method threshold: score >= 0.95 (not 0.9)")
        print("  2. Require single words only: word_count == 1 (not <= 2)")
        print("  3. Lower consensus thresholds: double consensus >= 0.4 (not 0.5)")
    elif without_cap > 70:
        print("\n⚠️ Natural filtering needs minor adjustment")
        print("\n💡 Try these tweaks:")
        print("  1. Single-method: score >= 0.92")
        print("  2. Strict word count: <= 1 for single method")
    else:
        print("\n✅ Natural filtering is close! Just needs fine-tuning")
        
elif without_cap <= 60:
    print("✅ Natural filtering is working well!")
    print(f"   Only {without_cap} concepts without hard cap")
    print("\n💡 The 50-cap is just a safety net - good job!")
else:
    print(f"🤷 Mixed results:")
    print(f"   With cap: {with_cap}")
    print(f"   Without cap: {without_cap}")

print("\n📊 Distribution comparison:")
if result1.get('purity_analysis') and result2.get('purity_analysis'):
    dist1 = result1['purity_analysis']['distribution']
    dist2 = result2['purity_analysis']['distribution']
    
    print("                     With Cap | Without Cap")
    print(f"  Consensus:         {dist1.get('consensus', 0):>8} | {dist2.get('consensus', 0):>8}")
    print(f"  High confidence:   {dist1.get('high_confidence', 0):>8} | {dist2.get('high_confidence', 0):>8}")
    print(f"  Database boosted:  {dist1.get('file_storage_boosted', 0):>8} | {dist2.get('file_storage_boosted', 0):>8}")
    print(f"  Single method:     {dist1.get('single_method', 0):>8} | {dist2.get('single_method', 0):>8}")

print("\n✅ Test complete! Pipeline restored to original state.")
