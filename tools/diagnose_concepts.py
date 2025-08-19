# tools/diagnose_concepts.py — Check concept count and metadata
"""
Diagnostic tool for analyzing concept extraction results.

This tool analyzes semantic_concepts.json files to verify concept count,
quality, and metadata completeness. It addresses Issue #5 from the triage
document by providing visibility into ingestion pipeline performance.

Usage:
    python diagnose_concepts.py path/to/semantic_concepts.json
    python diagnose_concepts.py --dir path/to/concept/outputs/
    python diagnose_concepts.py --compare before.json after.json

Features:
- Concept count and distribution analysis
- Confidence score statistics
- Metadata completeness checking
- Extraction method analysis
- Source reference validation
- Batch directory processing
- Before/after comparison mode
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import statistics

def load_concept_file(file_path: str) -> List[Dict[str, Any]]:
    """Load concepts from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "concepts" in data:
            return data["concepts"]
        else:
            print(f"⚠️  Warning: Unexpected JSON structure in {file_path}")
            return []
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []

def analyze_concepts(concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze a list of concepts and return statistics."""
    if not concepts:
        return {"error": "No concepts to analyze"}
    
    # Basic counts
    total_count = len(concepts)
    
    # Confidence analysis
    confidences = [c.get("confidence", 0.0) for c in concepts if isinstance(c.get("confidence"), (int, float))]
    confidence_stats = {}
    if confidences:
        confidence_stats = {
            "min": min(confidences),
            "max": max(confidences),
            "mean": statistics.mean(confidences),
            "median": statistics.median(confidences),
            "std_dev": statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        }
    
    # Method analysis
    methods = [c.get("method", "unknown") for c in concepts]
    method_counts = Counter(methods)
    
    # Source analysis
    sources = []
    for c in concepts:
        source = c.get("source", {})
        if isinstance(source, dict):
            if "page" in source:
                sources.append(f"page_{source['page']}")
            elif "segment" in source:
                sources.append(f"segment_{source['segment']}")
            else:
                sources.append("unknown_source")
        else:
            sources.append(str(source))
    source_counts = Counter(sources)
    
    # Metadata completeness
    required_fields = ["name", "confidence", "method", "source"]
    optional_fields = ["context", "embedding", "eigenfunction_id"]
    
    field_presence = {}
    for field in required_fields + optional_fields:
        present = sum(1 for c in concepts if field in c and c[field] is not None)
        field_presence[field] = {
            "count": present,
            "percentage": (present / total_count) * 100
        }
    
    # Quality tiers based on confidence
    quality_tiers = {"high": 0, "medium": 0, "low": 0, "missing": 0}
    for c in concepts:
        conf = c.get("confidence")
        if conf is None:
            quality_tiers["missing"] += 1
        elif conf >= 0.8:
            quality_tiers["high"] += 1
        elif conf >= 0.6:
            quality_tiers["medium"] += 1
        else:
            quality_tiers["low"] += 1
    
    return {
        "total_count": total_count,
        "confidence_stats": confidence_stats,
        "method_distribution": dict(method_counts),
        "source_distribution": dict(source_counts.most_common(10)),
        "field_presence": field_presence,
        "quality_tiers": quality_tiers
    }

def print_analysis(file_path: str, analysis: Dict[str, Any]):
    """Print formatted analysis results."""
    print(f"\n📄 Analyzing: {file_path}")
    print("=" * 80)
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    # Basic info
    print(f"✅ Total Concepts: {analysis['total_count']}")
    
    # Confidence statistics
    conf_stats = analysis.get("confidence_stats", {})
    if conf_stats:
        print(f"\n📊 Confidence Statistics:")
        print(f"   Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")
        print(f"   Mean:  {conf_stats['mean']:.3f} ± {conf_stats['std_dev']:.3f}")
        print(f"   Median: {conf_stats['median']:.3f}")
    
    # Quality distribution
    quality = analysis.get("quality_tiers", {})
    print(f"\n🏆 Quality Distribution:")
    print(f"   High (≥0.8):   {quality['high']:3d} concepts")
    print(f"   Medium (≥0.6): {quality['medium']:3d} concepts") 
    print(f"   Low (<0.6):    {quality['low']:3d} concepts")
    if quality['missing'] > 0:
        print(f"   Missing conf:  {quality['missing']:3d} concepts")
    
    # Method distribution
    methods = analysis.get("method_distribution", {})
    print(f"\n🔧 Extraction Methods:")
    for method, count in methods.items():
        percentage = (count / analysis['total_count']) * 100
        print(f"   {method:20s}: {count:3d} ({percentage:5.1f}%)")
    
    # Source distribution (top 5)
    sources = analysis.get("source_distribution", {})
    print(f"\n📍 Top Sources:")
    for source, count in list(sources.items())[:5]:
        percentage = (count / analysis['total_count']) * 100
        print(f"   {source:20s}: {count:3d} ({percentage:5.1f}%)")
    
    # Metadata completeness
    field_presence = analysis.get("field_presence", {})
    print(f"\n📋 Metadata Completeness:")
    required_fields = ["name", "confidence", "method", "source"]
    optional_fields = ["context", "embedding", "eigenfunction_id"]
    
    print("   Required Fields:")
    for field in required_fields:
        if field in field_presence:
            stats = field_presence[field]
            status = "✅" if stats["percentage"] >= 95 else "⚠️ " if stats["percentage"] >= 80 else "❌"
            print(f"   {status} {field:15s}: {stats['count']:3d}/{analysis['total_count']} ({stats['percentage']:5.1f}%)")
    
    print("   Optional Fields:")
    for field in optional_fields:
        if field in field_presence:
            stats = field_presence[field]
            status = "✅" if stats["percentage"] >= 50 else "📝"
            print(f"   {status} {field:15s}: {stats['count']:3d}/{analysis['total_count']} ({stats['percentage']:5.1f}%)")

def print_sample_concepts(concepts: List[Dict[str, Any]], count: int = 5):
    """Print sample concepts for inspection."""
    if not concepts:
        return
    
    print(f"\n🔍 Sample Concepts (showing {min(count, len(concepts))}):")
    print("-" * 80)
    
    for i, concept in enumerate(concepts[:count]):
        name = concept.get("name", "Unnamed")
        confidence = concept.get("confidence", "?")
        method = concept.get("method", "unknown")
        source = concept.get("source", {})
        context = concept.get("context", "")[:60] + "..." if concept.get("context") else "No context"
        
        print(f"  {i+1:2d}. {name}")
        print(f"      Confidence: {confidence:.3f} | Method: {method}")
        print(f"      Source: {source}")
        print(f"      Context: {context}")
        print()

def compare_concept_files(before_path: str, after_path: str):
    """Compare two concept files and show differences."""
    print(f"\n🔄 Comparing Concept Files")
    print("=" * 80)
    
    before_concepts = load_concept_file(before_path)
    after_concepts = load_concept_file(after_path)
    
    before_names = set(c.get("name", "") for c in before_concepts)
    after_names = set(c.get("name", "") for c in after_concepts)
    
    added = after_names - before_names
    removed = before_names - after_names
    common = before_names & after_names
    
    print(f"BEFORE: {before_path} → {len(before_concepts)} concepts")
    print(f"AFTER:  {after_path} → {len(after_concepts)} concepts")
    print(f"CHANGE: {len(after_concepts) - len(before_concepts):+d} concepts")
    print()
    
    if added:
        print(f"➕ Added Concepts ({len(added)}):")
        for name in sorted(added)[:10]:  # Show first 10
            print(f"   + {name}")
        if len(added) > 10:
            print(f"   ... and {len(added) - 10} more")
        print()
    
    if removed:
        print(f"➖ Removed Concepts ({len(removed)}):")
        for name in sorted(removed)[:10]:  # Show first 10
            print(f"   - {name}")
        if len(removed) > 10:
            print(f"   ... and {len(removed) - 10} more")
        print()
    
    print(f"✅ Unchanged Concepts: {len(common)}")
    
    # Compare quality metrics
    if before_concepts and after_concepts:
        before_analysis = analyze_concepts(before_concepts)
        after_analysis = analyze_concepts(after_concepts)
        
        before_conf = before_analysis.get("confidence_stats", {}).get("mean", 0)
        after_conf = after_analysis.get("confidence_stats", {}).get("mean", 0)
        
        print(f"\n📊 Quality Comparison:")
        print(f"   Mean Confidence: {before_conf:.3f} → {after_conf:.3f} ({after_conf - before_conf:+.3f})")
        
        before_high = before_analysis.get("quality_tiers", {}).get("high", 0)
        after_high = after_analysis.get("quality_tiers", {}).get("high", 0)
        print(f"   High Quality:    {before_high} → {after_high} ({after_high - before_high:+d})")

def process_directory(dir_path: str):
    """Process all concept JSON files in a directory."""
    directory = Path(dir_path)
    
    if not directory.exists():
        print(f"❌ Directory not found: {dir_path}")
        return
    
    # Find concept JSON files
    json_files = list(directory.glob("**/*concept*.json"))
    json_files.extend(directory.glob("**/*semantic*.json"))
    
    if not json_files:
        print(f"📂 No concept JSON files found in {dir_path}")
        return
    
    print(f"📂 Processing {len(json_files)} files in {dir_path}")
    print("=" * 80)
    
    total_concepts = 0
    successful_files = 0
    
    for file_path in sorted(json_files):
        concepts = load_concept_file(str(file_path))
        if concepts:
            total_concepts += len(concepts)
            successful_files += 1
            
            # Brief analysis
            analysis = analyze_concepts(concepts)
            conf_stats = analysis.get("confidence_stats", {})
            avg_conf = conf_stats.get("mean", 0)
            
            print(f"✅ {file_path.name:40s} {len(concepts):4d} concepts (avg conf: {avg_conf:.2f})")
        else:
            print(f"❌ {file_path.name:40s} failed to load or empty")
    
    print(f"\n📋 Summary: {successful_files}/{len(json_files)} files processed, {total_concepts} total concepts")

def main():
    parser = argparse.ArgumentParser(description="Diagnose concept extraction results")
    parser.add_argument("input", nargs="?", help="Path to concept JSON file or directory")
    parser.add_argument("--dir", help="Process all concept files in directory")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), 
                       help="Compare two concept files")
    parser.add_argument("--samples", type=int, default=5, 
                       help="Number of sample concepts to show (default: 5)")
    parser.add_argument("--no-samples", action="store_true", 
                       help="Skip showing sample concepts")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_concept_files(args.compare[0], args.compare[1])
    elif args.dir:
        process_directory(args.dir)
    elif args.input:
        if Path(args.input).is_dir():
            process_directory(args.input)
        else:
            concepts = load_concept_file(args.input)
            if concepts:
                analysis = analyze_concepts(concepts)
                print_analysis(args.input, analysis)
                
                if not args.no_samples:
                    print_sample_concepts(concepts, args.samples)
            else:
                print(f"❌ No concepts found in {args.input}")
    else:
        # Default behavior: look for semantic_concepts.json in current directory
        default_file = "semantic_concepts.json"
        if Path(default_file).exists():
            concepts = load_concept_file(default_file)
            if concepts:
                analysis = analyze_concepts(concepts)
                print_analysis(default_file, analysis)
                
                if not args.no_samples:
                    print_sample_concepts(concepts, args.samples)
        else:
            print("Usage: python diagnose_concepts.py <semantic_concepts.json>")
            print("       python diagnose_concepts.py --dir <directory>")
            print("       python diagnose_concepts.py --compare <before.json> <after.json>")

if __name__ == "__main__":
    main()
