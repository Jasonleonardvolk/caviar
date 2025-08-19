# tools/compare_ingestion_runs.py — Compare two semantic_concepts.json outputs
"""
Comparison tool for concept ingestion results.

This tool compares two concept extraction outputs to analyze the impact
of pipeline changes, configuration adjustments, or algorithm improvements.
It addresses Issue #5 from the triage document by providing quantitative
feedback on ingestion improvements.

Usage:
    python compare_ingestion_runs.py before.json after.json
    python compare_ingestion_runs.py --batch before_dir/ after_dir/
    python compare_ingestion_runs.py --baseline baseline.json current.json --save-report

Features:
- Concept count comparison
- Quality metric analysis
- Added/removed concept tracking
- Confidence score distribution changes
- Method and source distribution comparison
- Detailed statistical analysis
- Report generation
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import Counter, defaultdict
import statistics
from datetime import datetime

def load_concept_file(file_path: str) -> Tuple[List[Dict[str, Any]], str]:
    """Load concepts from JSON file and return concepts with metadata."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle different JSON structures
        concepts = []
        metadata = {}
        
        if isinstance(data, list):
            concepts = data
        elif isinstance(data, dict):
            if "concepts" in data:
                concepts = data["concepts"]
                metadata = {k: v for k, v in data.items() if k != "concepts"}
            else:
                # Assume the dict contains metadata and concepts mixed
                concepts = [v for v in data.values() if isinstance(v, dict) and "name" in v]
                metadata = {k: v for k, v in data.items() if not (isinstance(v, dict) and "name" in v)}
        
        return concepts, json.dumps(metadata) if metadata else ""
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return [], ""

def get_concept_names(concepts: List[Dict[str, Any]]) -> Set[str]:
    """Extract concept names from concept list."""
    return set(c.get("name", "") for c in concepts if c.get("name"))

def calculate_concept_stats(concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive statistics for a concept list."""
    if not concepts:
        return {"count": 0}
    
    # Basic counts
    total_count = len(concepts)
    
    # Confidence analysis
    confidences = [c.get("confidence", 0.0) for c in concepts if isinstance(c.get("confidence"), (int, float))]
    confidence_stats = {}
    if confidences:
        confidence_stats = {
            "count": len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "mean": statistics.mean(confidences),
            "median": statistics.median(confidences),
            "std_dev": statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        }
    
    # Quality distribution
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
    
    # Method distribution
    methods = Counter(c.get("method", "unknown") for c in concepts)
    
    # Source distribution
    sources = Counter()
    for c in concepts:
        source = c.get("source", {})
        if isinstance(source, dict):
            if "page" in source:
                sources[f"page_{source['page']}"] += 1
            elif "segment" in source:
                sources[f"segment_{source['segment']}"] += 1
            else:
                sources["other"] += 1
        else:
            sources[str(source)] += 1
    
    # Metadata completeness
    required_fields = ["name", "confidence", "method", "source"]
    optional_fields = ["context", "embedding", "eigenfunction_id"]
    
    field_completeness = {}
    for field in required_fields + optional_fields:
        present = sum(1 for c in concepts if field in c and c[field] is not None and c[field] != "")
        field_completeness[field] = present / total_count if total_count > 0 else 0
    
    return {
        "count": total_count,
        "confidence_stats": confidence_stats,
        "quality_tiers": quality_tiers,
        "method_distribution": dict(methods),
        "source_distribution": dict(sources),
        "field_completeness": field_completeness
    }

def compare_distributions(before_dist: Dict[str, int], after_dist: Dict[str, int]) -> Dict[str, Any]:
    """Compare two distributions and return change analysis."""
    all_keys = set(before_dist.keys()) | set(after_dist.keys())
    
    changes = {}
    added = {}
    removed = {}
    modified = {}
    
    for key in all_keys:
        before_val = before_dist.get(key, 0)
        after_val = after_dist.get(key, 0)
        
        if before_val == 0 and after_val > 0:
            added[key] = after_val
        elif before_val > 0 and after_val == 0:
            removed[key] = before_val
        elif before_val != after_val:
            modified[key] = {
                "before": before_val,
                "after": after_val,
                "change": after_val - before_val
            }
    
    return {
        "added": added,
        "removed": removed,
        "modified": modified
    }

def print_comparison_header(before_path: str, after_path: str):
    """Print comparison header."""
    print("\n🔄 Concept Ingestion Comparison")
    print("=" * 80)
    print(f"BEFORE: {before_path}")
    print(f"AFTER:  {after_path}")
    print(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def print_concept_changes(before_names: Set[str], after_names: Set[str]):
    """Print added/removed concept analysis."""
    added = after_names - before_names
    removed = before_names - after_names
    common = before_names & after_names
    
    print(f"\n📊 Concept Changes:")
    print(f"   Total Before: {len(before_names):4d}")
    print(f"   Total After:  {len(after_names):4d}")
    print(f"   Net Change:   {len(after_names) - len(before_names):+4d}")
    print(f"   Unchanged:    {len(common):4d}")
    print()
    
    if added:
        print(f"➕ Added Concepts ({len(added)}):")
        for name in sorted(list(added)[:10]):  # Show first 10
            print(f"   + {name}")
        if len(added) > 10:
            print(f"   ... and {len(added) - 10} more")
        print()
    
    if removed:
        print(f"➖ Removed Concepts ({len(removed)}):")
        for name in sorted(list(removed)[:10]):  # Show first 10
            print(f"   - {name}")
        if len(removed) > 10:
            print(f"   ... and {len(removed) - 10} more")
        print()

def print_quality_comparison(before_stats: Dict[str, Any], after_stats: Dict[str, Any]):
    """Print quality metrics comparison."""
    print("\n📈 Quality Metrics Comparison:")
    
    # Confidence statistics
    before_conf = before_stats.get("confidence_stats", {})
    after_conf = after_stats.get("confidence_stats", {})
    
    if before_conf and after_conf:
        print("   Confidence Scores:")
        metrics = ["mean", "median", "min", "max", "std_dev"]
        for metric in metrics:
            if metric in before_conf and metric in after_conf:
                before_val = before_conf[metric]
                after_val = after_conf[metric]
                change = after_val - before_val
                symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"   {symbol} {metric.title():8s}: {before_val:.3f} → {after_val:.3f} ({change:+.3f})")
    
    # Quality tiers
    before_quality = before_stats.get("quality_tiers", {})
    after_quality = after_stats.get("quality_tiers", {})
    
    if before_quality and after_quality:
        print("\n   Quality Distribution:")
        tiers = ["high", "medium", "low", "missing"]
        for tier in tiers:
            before_val = before_quality.get(tier, 0)
            after_val = after_quality.get(tier, 0)
            change = after_val - before_val
            symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            
            # Calculate percentages
            before_pct = (before_val / before_stats["count"]) * 100 if before_stats["count"] > 0 else 0
            after_pct = (after_val / after_stats["count"]) * 100 if after_stats["count"] > 0 else 0
            
            print(f"   {symbol} {tier.title():8s}: {before_val:3d} ({before_pct:4.1f}%) → {after_val:3d} ({after_pct:4.1f}%) ({change:+3d})")

def print_distribution_comparison(title: str, before_dist: Dict[str, int], after_dist: Dict[str, int]):
    """Print distribution comparison."""
    changes = compare_distributions(before_dist, after_dist)
    
    print(f"\n🔧 {title} Distribution Changes:")
    
    if changes["added"]:
        print("   Added:")
        for key, count in sorted(changes["added"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   + {key:20s}: {count:3d}")
    
    if changes["removed"]:
        print("   Removed:")
        for key, count in sorted(changes["removed"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   - {key:20s}: {count:3d}")
    
    if changes["modified"]:
        print("   Modified:")
        for key, change_data in sorted(changes["modified"].items(), 
                                     key=lambda x: abs(x[1]["change"]), reverse=True)[:5]:
            change_val = change_data["change"]
            symbol = "📈" if change_val > 0 else "📉"
            print(f"   {symbol} {key:20s}: {change_data['before']:3d} → {change_data['after']:3d} ({change_val:+3d})")

def print_completeness_comparison(before_stats: Dict[str, Any], after_stats: Dict[str, Any]):
    """Print metadata completeness comparison."""
    before_comp = before_stats.get("field_completeness", {})
    after_comp = after_stats.get("field_completeness", {})
    
    if not before_comp or not after_comp:
        return
    
    print("\n📋 Metadata Completeness Changes:")
    
    required_fields = ["name", "confidence", "method", "source"]
    optional_fields = ["context", "embedding", "eigenfunction_id"]
    
    print("   Required Fields:")
    for field in required_fields:
        if field in before_comp and field in after_comp:
            before_pct = before_comp[field] * 100
            after_pct = after_comp[field] * 100
            change = after_pct - before_pct
            symbol = "✅" if after_pct >= 95 else "⚠️ " if after_pct >= 80 else "❌"
            trend = "📈" if change > 1 else "📉" if change < -1 else "➡️"
            print(f"   {symbol}{trend} {field:15s}: {before_pct:5.1f}% → {after_pct:5.1f}% ({change:+5.1f}%)")
    
    print("   Optional Fields:")
    for field in optional_fields:
        if field in before_comp and field in after_comp:
            before_pct = before_comp[field] * 100
            after_pct = after_comp[field] * 100
            change = after_pct - before_pct
            symbol = "✅" if after_pct >= 50 else "📝"
            trend = "📈" if change > 1 else "📉" if change < -1 else "➡️"
            print(f"   {symbol}{trend} {field:15s}: {before_pct:5.1f}% → {after_pct:5.1f}% ({change:+5.1f}%)")

def generate_comparison_report(
    before_path: str, 
    after_path: str, 
    before_concepts: List[Dict[str, Any]], 
    after_concepts: List[Dict[str, Any]],
    output_path: str = None
) -> str:
    """Generate a detailed comparison report."""
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"comparison_report_{timestamp}.md"
    
    before_stats = calculate_concept_stats(before_concepts)
    after_stats = calculate_concept_stats(after_concepts)
    before_names = get_concept_names(before_concepts)
    after_names = get_concept_names(after_concepts)
    
    report = []
    report.append("# Concept Ingestion Comparison Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Before:** {before_path}")
    report.append(f"**After:** {after_path}")
    report.append("")
    
    # Summary
    report.append("## Summary")
    report.append("")
    report.append(f"- **Concepts Before:** {len(before_concepts)}")
    report.append(f"- **Concepts After:** {len(after_concepts)}")
    report.append(f"- **Net Change:** {len(after_concepts) - len(before_concepts):+d}")
    report.append(f"- **Added:** {len(after_names - before_names)}")
    report.append(f"- **Removed:** {len(before_names - after_names)}")
    report.append(f"- **Unchanged:** {len(before_names & after_names)}")
    report.append("")
    
    # Quality metrics
    if before_stats.get("confidence_stats") and after_stats.get("confidence_stats"):
        before_conf = before_stats["confidence_stats"]
        after_conf = after_stats["confidence_stats"]
        
        report.append("## Quality Metrics")
        report.append("")
        report.append("| Metric | Before | After | Change |")
        report.append("|--------|--------|-------|---------|")
        
        metrics = ["mean", "median", "min", "max"]
        for metric in metrics:
            if metric in before_conf and metric in after_conf:
                before_val = before_conf[metric]
                after_val = after_conf[metric]
                change = after_val - before_val
                report.append(f"| {metric.title()} | {before_val:.3f} | {after_val:.3f} | {change:+.3f} |")
        report.append("")
    
    # Save report
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"\n📄 Detailed report saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Failed to save report: {e}")
        return ""

def compare_files(before_path: str, after_path: str, save_report: bool = False):
    """Compare two concept files."""
    # Load concept files
    before_concepts, before_meta = load_concept_file(before_path)
    after_concepts, after_meta = load_concept_file(after_path)
    
    if not before_concepts and not after_concepts:
        print("❌ Both files are empty or failed to load")
        return
    
    # Print comparison
    print_comparison_header(before_path, after_path)
    
    # Basic concept changes
    before_names = get_concept_names(before_concepts)
    after_names = get_concept_names(after_concepts)
    print_concept_changes(before_names, after_names)
    
    # Quality comparison
    if before_concepts and after_concepts:
        before_stats = calculate_concept_stats(before_concepts)
        after_stats = calculate_concept_stats(after_concepts)
        
        print_quality_comparison(before_stats, after_stats)
        print_distribution_comparison("Method", 
                                    before_stats.get("method_distribution", {}),
                                    after_stats.get("method_distribution", {}))
        print_distribution_comparison("Source",
                                    before_stats.get("source_distribution", {}),
                                    after_stats.get("source_distribution", {}))
        print_completeness_comparison(before_stats, after_stats)
        
        # Generate report if requested
        if save_report:
            generate_comparison_report(before_path, after_path, before_concepts, after_concepts)

def compare_directories(before_dir: str, after_dir: str):
    """Compare all concept files in two directories."""
    before_path = Path(before_dir)
    after_path = Path(after_dir)
    
    if not before_path.exists() or not after_path.exists():
        print("❌ One or both directories do not exist")
        return
    
    # Find matching files
    before_files = {f.name: f for f in before_path.glob("**/*concept*.json")}
    before_files.update({f.name: f for f in before_path.glob("**/*semantic*.json")})
    
    after_files = {f.name: f for f in after_path.glob("**/*concept*.json")}
    after_files.update({f.name: f for f in after_path.glob("**/*semantic*.json")})
    
    common_files = set(before_files.keys()) & set(after_files.keys())
    
    if not common_files:
        print("❌ No matching concept files found in both directories")
        return
    
    print(f"\n📂 Comparing {len(common_files)} matching files")
    print("=" * 80)
    
    total_before = 0
    total_after = 0
    
    for filename in sorted(common_files):
        before_concepts, _ = load_concept_file(str(before_files[filename]))
        after_concepts, _ = load_concept_file(str(after_files[filename]))
        
        before_count = len(before_concepts)
        after_count = len(after_concepts)
        change = after_count - before_count
        
        total_before += before_count
        total_after += after_count
        
        symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        print(f"{symbol} {filename:40s} {before_count:4d} → {after_count:4d} ({change:+4d})")
    
    print("=" * 80)
    print(f"📊 Total: {total_before:4d} → {total_after:4d} ({total_after - total_before:+4d})")

def main():
    parser = argparse.ArgumentParser(description="Compare concept ingestion results")
    parser.add_argument("before", help="Before concept file or directory")
    parser.add_argument("after", help="After concept file or directory")
    parser.add_argument("--batch", action="store_true", 
                       help="Compare directories instead of files")
    parser.add_argument("--save-report", action="store_true",
                       help="Save detailed comparison report")
    parser.add_argument("--baseline", action="store_true",
                       help="Treat first file as baseline for comparison")
    
    args = parser.parse_args()
    
    if args.batch or (Path(args.before).is_dir() and Path(args.after).is_dir()):
        compare_directories(args.before, args.after)
    else:
        compare_files(args.before, args.after, args.save_report)

if __name__ == "__main__":
    main()
