#!/usr/bin/env python3
"""
📊 PRODUCTION MONITORING - Track Pipeline Performance
"""

import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

class ProductionMonitor:
    def __init__(self):
        self.log_file = Path("production_metrics.json")
        self.load_metrics()
    
    def load_metrics(self):
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {
                "deployments": [],
                "performance_history": [],
                "concept_quality": [],
                "user_satisfaction": []
            }
    
    def save_metrics(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_extraction(self, result):
        """Log metrics from an extraction"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "filename": result.get("filename", "unknown"),
            "file_size_mb": result.get("file_size_mb", 0),
            "processing_time": result.get("processing_time_seconds", 0),
            "chunks_processed": result.get("chunks_processed", 0),
            "chunks_available": result.get("chunks_available", 0),
            "raw_concepts": result.get("purity_analysis", {}).get("raw_concepts", 0),
            "final_concepts": result.get("concept_count", 0),
            "purity_efficiency": result.get("purity_analysis", {}).get("purity_efficiency", "0%"),
            "consensus_concepts": result.get("purity_analysis", {}).get("distribution", {}).get("consensus", 0)
        }
        
        self.metrics["performance_history"].append(metric)
        self.save_metrics()
        
        # Print immediate feedback
        print(f"\n📊 EXTRACTION METRICS:")
        print(f"  File: {metric['filename']}")
        print(f"  Time: {metric['processing_time']:.1f}s")
        print(f"  Concepts: {metric['raw_concepts']} → {metric['final_concepts']}")
        print(f"  Efficiency: {metric['purity_efficiency']}")
    
    def generate_report(self):
        """Generate performance report"""
        if not self.metrics["performance_history"]:
            print("No metrics collected yet!")
            return
        
        print("\n" + "="*70)
        print("📊 PRODUCTION PERFORMANCE REPORT")
        print("="*70)
        
        # Calculate averages
        recent = self.metrics["performance_history"][-20:]  # Last 20 extractions
        
        avg_time = sum(m["processing_time"] for m in recent) / len(recent)
        avg_concepts = sum(m["final_concepts"] for m in recent) / len(recent)
        avg_efficiency = sum(float(m["purity_efficiency"].rstrip('%')) for m in recent) / len(recent)
        
        print(f"\n📈 AVERAGES (Last {len(recent)} extractions):")
        print(f"  Processing time: {avg_time:.1f}s")
        print(f"  Concepts per doc: {avg_concepts:.1f}")
        print(f"  Purity efficiency: {avg_efficiency:.1f}%")
        
        # File size analysis
        size_buckets = defaultdict(list)
        for m in recent:
            size = m.get("file_size_mb", 0)
            if size < 1:
                size_buckets["small"].append(m)
            elif size < 5:
                size_buckets["medium"].append(m)
            else:
                size_buckets["large"].append(m)
        
        print(f"\n📏 PERFORMANCE BY FILE SIZE:")
        for size, metrics in size_buckets.items():
            if metrics:
                avg_time = sum(m["processing_time"] for m in metrics) / len(metrics)
                avg_chunks = sum(m["chunks_processed"] for m in metrics) / len(metrics)
                print(f"  {size.capitalize()} files: {avg_time:.1f}s, {avg_chunks:.1f} chunks")
        
        # Improvement tracking
        if len(self.metrics["performance_history"]) > 10:
            old = self.metrics["performance_history"][:10]
            new = self.metrics["performance_history"][-10:]
            
            old_time = sum(m["processing_time"] for m in old) / len(old)
            new_time = sum(m["processing_time"] for m in new) / len(new)
            
            improvement = ((old_time - new_time) / old_time) * 100
            
            print(f"\n🚀 PERFORMANCE IMPROVEMENT:")
            print(f"  Old avg: {old_time:.1f}s")
            print(f"  New avg: {new_time:.1f}s")
            print(f"  Improvement: {improvement:.1f}%")
        
        print("\n✅ Report generated!")
        self.save_metrics()

# Usage
if __name__ == "__main__":
    monitor = ProductionMonitor()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        monitor.generate_report()
    else:
        print("📊 Production Monitor Ready!")
        print("Usage:")
        print("  python production_monitor.py report  # Generate report")
        print("\nTo log metrics, import and use:")
        print("  from production_monitor import ProductionMonitor")
        print("  monitor = ProductionMonitor()")
        print("  monitor.log_extraction(result)")
