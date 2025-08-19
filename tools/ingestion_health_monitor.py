# tools/ingestion_health_monitor.py — Ingestion QA Sweep
"""
Production health monitoring tool for concept ingestion pipeline.

This tool provides continuous monitoring of concept ingestion quality,
scanning output files for anomalies, broken schemas, and performance issues.
It addresses Issue #5 from the triage document by providing production
quality assurance monitoring.

Usage:
    python ingestion_health_monitor.py
    python ingestion_health_monitor.py --dir ./concept_outputs/
    python ingestion_health_monitor.py --continuous --interval 300
    python ingestion_health_monitor.py --alert-webhook https://alerts.example.com/hook

Features:
- Batch health scanning of concept files
- Anomaly detection in concept distributions
- Schema validation and completeness checking
- Performance trend analysis
- Alert integration for production monitoring
- Continuous monitoring mode
- Health dashboard generation
"""

import os
import json
import time
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import statistics
import hashlib

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class HealthIssue:
    """Represents a health issue found during monitoring."""
    
    SEVERITY_CRITICAL = "CRITICAL"
    SEVERITY_WARNING = "WARNING"
    SEVERITY_INFO = "INFO"
    
    def __init__(self, severity: str, category: str, message: str, details: Optional[Dict] = None):
        self.severity = severity
        self.category = category
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        self.file_path = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "file_path": self.file_path
        }
    
    def __str__(self) -> str:
        severity_icons = {
            self.SEVERITY_CRITICAL: "🚨",
            self.SEVERITY_WARNING: "⚠️ ",
            self.SEVERITY_INFO: "ℹ️ "
        }
        icon = severity_icons.get(self.severity, "📝")
        return f"{icon} [{self.severity}] {self.category}: {self.message}"

class ConceptFileAnalyzer:
    """Analyzes individual concept files for health issues."""
    
    def __init__(self):
        self.required_fields = ["name", "confidence", "method", "source"]
        self.recommended_fields = ["context", "embedding", "eigenfunction_id"]
        
        # Health thresholds
        self.min_concepts_threshold = 1
        self.max_concepts_threshold = 100
        self.min_confidence_threshold = 0.3
        self.min_completeness_threshold = 0.8
    
    def analyze_file(self, file_path: Path) -> Tuple[Dict[str, Any], List[HealthIssue]]:
        """Analyze a concept file and return metrics + issues."""
        issues = []
        metrics = {
            "file_path": str(file_path),
            "file_size": 0,
            "last_modified": None,
            "concept_count": 0,
            "load_successful": False
        }
        
        try:
            # Basic file info
            stat = file_path.stat()
            metrics["file_size"] = stat.st_size
            metrics["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            # Load and parse JSON
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    issues.append(HealthIssue(
                        HealthIssue.SEVERITY_CRITICAL,
                        "empty_file",
                        f"File is empty: {file_path.name}"
                    ))
                    return metrics, issues
                
                data = json.loads(content)
            
            metrics["load_successful"] = True
            
            # Extract concepts from different JSON structures
            concepts = self._extract_concepts(data)
            metrics["concept_count"] = len(concepts)
            
            # Check concept count
            if len(concepts) == 0:
                issues.append(HealthIssue(
                    HealthIssue.SEVERITY_CRITICAL,
                    "no_concepts",
                    f"No concepts extracted: {file_path.name}"
                ))
                return metrics, issues
            
            if len(concepts) > self.max_concepts_threshold:
                issues.append(HealthIssue(
                    HealthIssue.SEVERITY_WARNING,
                    "too_many_concepts",
                    f"Unusually high concept count: {len(concepts)} in {file_path.name}",
                    {"concept_count": len(concepts), "threshold": self.max_concepts_threshold}
                ))
            
            # Analyze concept quality
            quality_metrics = self._analyze_concept_quality(concepts)
            metrics.update(quality_metrics)
            
            # Check for quality issues
            quality_issues = self._check_quality_issues(concepts, file_path.name)
            issues.extend(quality_issues)
            
        except json.JSONDecodeError as e:
            issues.append(HealthIssue(
                HealthIssue.SEVERITY_CRITICAL,
                "json_corruption",
                f"JSON parsing failed: {file_path.name}",
                {"error": str(e)}
            ))
        except Exception as e:
            issues.append(HealthIssue(
                HealthIssue.SEVERITY_CRITICAL,
                "file_error",
                f"File analysis failed: {file_path.name}",
                {"error": str(e)}
            ))
        
        # Add file path to all issues
        for issue in issues:
            issue.file_path = str(file_path)
        
        return metrics, issues
    
    def _extract_concepts(self, data: Any) -> List[Dict[str, Any]]:
        """Extract concept list from various JSON structures."""
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict) and "name" in item]
        elif isinstance(data, dict):
            if "concepts" in data:
                return data["concepts"]
            # Look for concept-like objects in the dictionary
            concepts = []
            for value in data.values():
                if isinstance(value, list):
                    concepts.extend([item for item in value if isinstance(item, dict) and "name" in item])
                elif isinstance(value, dict) and "name" in value:
                    concepts.append(value)
            return concepts
        return []
    
    def _analyze_concept_quality(self, concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze concept quality metrics."""
        if not concepts:
            return {}
        
        # Confidence analysis
        confidences = [c.get("confidence", 0.0) for c in concepts if isinstance(c.get("confidence"), (int, float))]
        confidence_stats = {}
        if confidences:
            confidence_stats = {
                "confidence_min": min(confidences),
                "confidence_max": max(confidences),
                "confidence_mean": statistics.mean(confidences),
                "confidence_median": statistics.median(confidences),
                "confidence_std": statistics.stdev(confidences) if len(confidences) > 1 else 0.0
            }
        
        # Field completeness
        field_completeness = {}
        for field in self.required_fields + self.recommended_fields:
            present = sum(1 for c in concepts if field in c and c[field] is not None and c[field] != "")
            field_completeness[f"{field}_completeness"] = present / len(concepts)
        
        # Method distribution
        methods = Counter(c.get("method", "unknown") for c in concepts)
        
        # Source distribution
        sources = Counter()
        for c in concepts:
            source = c.get("source", {})
            if isinstance(source, dict):
                source_key = f"page_{source.get('page', '?')}" if "page" in source else "other"
            else:
                source_key = "other"
            sources[source_key] += 1
        
        return {
            **confidence_stats,
            **field_completeness,
            "method_diversity": len(methods),
            "source_diversity": len(sources),
            "most_common_method": methods.most_common(1)[0][0] if methods else "unknown",
            "most_common_source": sources.most_common(1)[0][0] if sources else "unknown"
        }
    
    def _check_quality_issues(self, concepts: List[Dict[str, Any]], filename: str) -> List[HealthIssue]:
        """Check for specific quality issues in concepts."""
        issues = []
        
        # Check confidence scores
        low_confidence_count = sum(1 for c in concepts 
                                 if isinstance(c.get("confidence"), (int, float)) and 
                                    c["confidence"] < self.min_confidence_threshold)
        
        if low_confidence_count > len(concepts) * 0.5:  # More than 50% low confidence
            issues.append(HealthIssue(
                HealthIssue.SEVERITY_WARNING,
                "low_confidence",
                f"High proportion of low-confidence concepts in {filename}",
                {"low_confidence_count": low_confidence_count, "total": len(concepts)}
            ))
        
        # Check missing confidence scores
        missing_confidence = sum(1 for c in concepts if "confidence" not in c or c["confidence"] is None)
        if missing_confidence > 0:
            issues.append(HealthIssue(
                HealthIssue.SEVERITY_WARNING,
                "missing_confidence",
                f"Concepts missing confidence scores in {filename}",
                {"missing_count": missing_confidence, "total": len(concepts)}
            ))
        
        # Check metadata completeness
        for field in self.required_fields:
            missing_count = sum(1 for c in concepts if field not in c or c[field] is None or c[field] == "")
            if missing_count / len(concepts) > (1 - self.min_completeness_threshold):
                issues.append(HealthIssue(
                    HealthIssue.SEVERITY_WARNING,
                    "incomplete_metadata",
                    f"High proportion of concepts missing {field} in {filename}",
                    {"field": field, "missing_count": missing_count, "total": len(concepts)}
                ))
        
        # Check for duplicate concept names
        names = [c.get("name", "") for c in concepts if c.get("name")]
        name_counts = Counter(names)
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        if duplicates:
            issues.append(HealthIssue(
                HealthIssue.SEVERITY_INFO,
                "duplicate_names",
                f"Duplicate concept names in {filename}",
                {"duplicates": duplicates}
            ))
        
        # Check for suspicious patterns
        methods = [c.get("method", "") for c in concepts]
        if len(set(methods)) == 1 and len(concepts) > 5:
            issues.append(HealthIssue(
                HealthIssue.SEVERITY_INFO,
                "method_uniformity",
                f"All concepts use same method in {filename}",
                {"method": methods[0], "count": len(concepts)}
            ))
        
        return issues

class IngestionHealthMonitor:
    """Main health monitoring class."""
    
    def __init__(self, base_dir: str = "./concept_outputs/"):
        self.base_dir = Path(base_dir)
        self.analyzer = ConceptFileAnalyzer()
        self.webhook_url = None
        self.last_scan_time = None
        self.scan_history = []
    
    def set_webhook(self, webhook_url: str):
        """Set webhook URL for alerts."""
        self.webhook_url = webhook_url
    
    def find_concept_files(self, since: Optional[datetime] = None) -> List[Path]:
        """Find concept JSON files, optionally filtered by modification time."""
        if not self.base_dir.exists():
            return []
        
        patterns = ["**/*concept*.json", "**/*semantic*.json"]
        files = []
        
        for pattern in patterns:
            found_files = list(self.base_dir.glob(pattern))
            for file_path in found_files:
                if since is None:
                    files.append(file_path)
                else:
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mod_time > since:
                        files.append(file_path)
        
        return sorted(set(files), key=lambda f: f.stat().st_mtime, reverse=True)
    
    def scan_health(self, files: Optional[List[Path]] = None) -> Dict[str, Any]:
        """Perform health scan on concept files."""
        if files is None:
            files = self.find_concept_files()
        
        if not files:
            return {
                "timestamp": datetime.now().isoformat(),
                "files_scanned": 0,
                "total_concepts": 0,
                "issues": [],
                "summary": {"critical": 0, "warning": 0, "info": 0}
            }
        
        print(f"🔍 Scanning {len(files)} concept files...")
        
        all_issues = []
        all_metrics = []
        total_concepts = 0
        
        for file_path in files:
            metrics, issues = self.analyzer.analyze_file(file_path)
            all_metrics.append(metrics)
            all_issues.extend(issues)
            total_concepts += metrics.get("concept_count", 0)
        
        # Categorize issues
        issue_summary = {"critical": 0, "warning": 0, "info": 0}
        for issue in all_issues:
            if issue.severity == HealthIssue.SEVERITY_CRITICAL:
                issue_summary["critical"] += 1
            elif issue.severity == HealthIssue.SEVERITY_WARNING:
                issue_summary["warning"] += 1
            else:
                issue_summary["info"] += 1
        
        scan_result = {
            "timestamp": datetime.now().isoformat(),
            "files_scanned": len(files),
            "total_concepts": total_concepts,
            "issues": [issue.to_dict() for issue in all_issues],
            "metrics": all_metrics,
            "summary": issue_summary
        }
        
        self.last_scan_time = datetime.now()
        self.scan_history.append(scan_result)
        
        # Keep only last 10 scans in memory
        if len(self.scan_history) > 10:
            self.scan_history = self.scan_history[-10:]
        
        return scan_result
    
    def print_scan_results(self, scan_result: Dict[str, Any]):
        """Print formatted scan results."""
        summary = scan_result["summary"]
        timestamp = scan_result["timestamp"]
        
        print(f"\n📊 Health Scan Results - {timestamp}")
        print("=" * 80)
        print(f"Files Scanned: {scan_result['files_scanned']}")
        print(f"Total Concepts: {scan_result['total_concepts']}")
        print(f"Issues Found: {sum(summary.values())}")
        print(f"  🚨 Critical: {summary['critical']}")
        print(f"  ⚠️  Warning:  {summary['warning']}")
        print(f"  ℹ️  Info:     {summary['info']}")
        
        # Print issues by category
        if scan_result["issues"]:
            print("\n📋 Issues by Category:")
            issues_by_category = defaultdict(list)
            for issue_data in scan_result["issues"]:
                issues_by_category[issue_data["category"]].append(issue_data)
            
            for category, category_issues in issues_by_category.items():
                print(f"\n  {category.upper().replace('_', ' ')} ({len(category_issues)} issues):")
                for issue_data in category_issues[:3]:  # Show first 3 of each type
                    severity_icons = {"CRITICAL": "🚨", "WARNING": "⚠️ ", "INFO": "ℹ️ "}
                    icon = severity_icons.get(issue_data["severity"], "📝")
                    filename = Path(issue_data["file_path"]).name
                    print(f"    {icon} {filename}: {issue_data['message']}")
                
                if len(category_issues) > 3:
                    print(f"    ... and {len(category_issues) - 3} more")
        
        # Overall health status
        if summary["critical"] > 0:
            print(f"\n🚨 HEALTH STATUS: CRITICAL - {summary['critical']} critical issues require immediate attention")
        elif summary["warning"] > 5:
            print(f"\n⚠️  HEALTH STATUS: DEGRADED - {summary['warning']} warnings detected")
        elif summary["warning"] > 0:
            print(f"\n⚠️  HEALTH STATUS: MINOR ISSUES - {summary['warning']} warnings detected")
        else:
            print(f"\n✅ HEALTH STATUS: GOOD - No critical issues detected")
    
    def send_alert(self, scan_result: Dict[str, Any]):
        """Send alert via webhook if issues are found."""
        if not self.webhook_url or not REQUESTS_AVAILABLE:
            return False
        
        summary = scan_result["summary"]
        if summary["critical"] == 0 and summary["warning"] == 0:
            return True  # No alerts needed
        
        # Prepare alert payload
        alert_data = {
            "timestamp": scan_result["timestamp"],
            "service": "TORI Concept Ingestion",
            "status": "CRITICAL" if summary["critical"] > 0 else "WARNING",
            "summary": f"{summary['critical']} critical, {summary['warning']} warning issues",
            "details": {
                "files_scanned": scan_result["files_scanned"],
                "total_concepts": scan_result["total_concepts"],
                "issues": scan_result["issues"][:10]  # Send first 10 issues
            }
        }
        
        try:
            response = requests.post(self.webhook_url, json=alert_data, timeout=10)
            if response.status_code < 300:
                print(f"📡 Alert sent successfully to webhook")
                return True
            else:
                print(f"⚠️  Alert webhook returned {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")
            return False
    
    def save_report(self, scan_result: Dict[str, Any], output_path: str = None):
        """Save detailed health report."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"health_report_{timestamp}.json"
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(scan_result, f, indent=2)
            print(f"📄 Health report saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
            return False
    
    def continuous_monitor(self, interval_seconds: int = 300):
        """Run continuous monitoring with specified interval."""
        print(f"🔄 Starting continuous monitoring (interval: {interval_seconds}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Look for files modified since last scan
                files_to_scan = None
                if self.last_scan_time:
                    files_to_scan = self.find_concept_files(since=self.last_scan_time)
                    if not files_to_scan:
                        print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - No new files to scan")
                        time.sleep(interval_seconds)
                        continue
                
                scan_result = self.scan_health(files_to_scan)
                self.print_scan_results(scan_result)
                
                # Send alerts if needed
                if scan_result["summary"]["critical"] > 0 or scan_result["summary"]["warning"] > 0:
                    self.send_alert(scan_result)
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description="Monitor concept ingestion health")
    parser.add_argument("--dir", default="./concept_outputs/",
                       help="Directory to monitor for concept files")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=300,
                       help="Monitoring interval in seconds (default: 300)")
    parser.add_argument("--alert-webhook", 
                       help="Webhook URL for sending alerts")
    parser.add_argument("--save-report", 
                       help="Save detailed report to file")
    parser.add_argument("--since-hours", type=int,
                       help="Only scan files modified in last N hours")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = IngestionHealthMonitor(args.dir)
    
    if args.alert_webhook:
        monitor.set_webhook(args.alert_webhook)
    
    if args.continuous:
        monitor.continuous_monitor(args.interval)
    else:
        # Single scan
        files = None
        if args.since_hours:
            since = datetime.now() - timedelta(hours=args.since_hours)
            files = monitor.find_concept_files(since=since)
            print(f"🔍 Scanning files modified in last {args.since_hours} hours")
        
        scan_result = monitor.scan_health(files)
        monitor.print_scan_results(scan_result)
        
        # Send alert if requested
        if args.alert_webhook:
            monitor.send_alert(scan_result)
        
        # Save report if requested
        if args.save_report:
            monitor.save_report(scan_result, args.save_report)

if __name__ == "__main__":
    main()
