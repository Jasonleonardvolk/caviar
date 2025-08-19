#!/usr/bin/env python3
"""
Prometheus Alert Overlap Checker

This script analyzes Prometheus alerting rules to detect overlaps in conditions,
which could lead to redundant or conflicting alerts. It parses all rule files
in the prometheus/rules directory and reports any rules with similar conditions.

Usage:
    python alert_overlap_checker.py [--threshold 0.8] [--report-file overlap_report.md]
"""

import os
import re
import sys
import yaml
import json
import glob
import logging
import argparse
import difflib
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("alert_overlap.log")
    ]
)
logger = logging.getLogger("alert_overlap")

class AlertRule:
    """Represents a Prometheus alert rule with its properties"""
    
    def __init__(self, name: str, expr: str, file: str, 
                 for_duration: str = "", labels: Dict[str, str] = None,
                 annotations: Dict[str, str] = None):
        self.name = name
        self.expr = expr
        self.file = file
        self.for_duration = for_duration
        self.labels = labels or {}
        self.annotations = annotations or {}
        
        # Normalize expression for comparison
        self.normalized_expr = self._normalize_expr(expr)
        
        # Extract metrics used in the expression
        self.metrics = self._extract_metrics(expr)
    
    def _normalize_expr(self, expr: str) -> str:
        """Normalize a PromQL expression for comparison"""
        # Remove comments
        expr = re.sub(r'#.*$', '', expr, flags=re.MULTILINE)
        
        # Remove extra whitespace
        expr = re.sub(r'\s+', ' ', expr).strip()
        
        # Normalize comparison operators
        expr = re.sub(r'\s*([=!<>]=?)\s*', r' \1 ', expr)
        
        # Normalize parentheses
        expr = re.sub(r'\(\s+', '(', expr)
        expr = re.sub(r'\s+\)', ')', expr)
        
        return expr
    
    def _extract_metrics(self, expr: str) -> Set[str]:
        """Extract metric names from a PromQL expression"""
        # This is a simplified approach - a full parser would be more accurate
        metrics = set()
        
        # Match patterns like 'metric_name{label="value"}' or just 'metric_name'
        pattern = r'([a-zA-Z_:][a-zA-Z0-9_:]*)\s*({[^}]*})?'
        matches = re.finditer(pattern, expr)
        
        for match in matches:
            metric = match.group(1)
            # Skip PromQL keywords and functions
            if metric not in {'and', 'or', 'unless', 'sum', 'avg', 'count', 'max', 'min',
                              'rate', 'increase', 'irate', 'by', 'on', 'group_right',
                              'group_left', 'offset', 'ignoring', 'without', 'histogram_quantile',
                              'label_replace', 'vector', 'scalar', 'absent', 'present',
                              'time', 'minute', 'hour', 'day', 'month', 'year', 'predict_linear'}:
                metrics.add(metric)
        
        return metrics
    
    def similarity_score(self, other: 'AlertRule') -> float:
        """Calculate similarity between this rule and another rule"""
        # Different approaches to similarity:
        
        # 1. Expression text similarity
        expr_similarity = difflib.SequenceMatcher(
            None, self.normalized_expr, other.normalized_expr
        ).ratio()
        
        # 2. Metrics overlap
        metrics_similarity = 0.0
        if self.metrics and other.metrics:
            common_metrics = self.metrics.intersection(other.metrics)
            all_metrics = self.metrics.union(other.metrics)
            metrics_similarity = len(common_metrics) / len(all_metrics) if all_metrics else 0.0
        
        # 3. For duration similarity (if applicable)
        duration_similarity = 1.0
        if self.for_duration and other.for_duration and self.for_duration != other.for_duration:
            duration_similarity = 0.5
        
        # Weighted average: expression has highest weight
        return 0.6 * expr_similarity + 0.3 * metrics_similarity + 0.1 * duration_similarity
    
    def __repr__(self) -> str:
        return f"AlertRule(name='{self.name}', file='{self.file}')"
    
    def details(self) -> str:
        """Get details about the rule for reporting"""
        return (
            f"**{self.name}**\n"
            f"- File: `{self.file}`\n"
            f"- Expression: `{self.expr}`\n"
            f"- For: {self.for_duration or 'not specified'}\n"
            f"- Severity: {self.labels.get('severity', 'not specified')}\n"
        )

def parse_rule_files(rules_dir: str = "prometheus/rules") -> List[AlertRule]:
    """Parse all Prometheus rule files in the specified directory"""
    all_rules = []
    
    # Find all YAML rule files
    rule_files = glob.glob(os.path.join(rules_dir, "*.yml")) + glob.glob(os.path.join(rules_dir, "*.yaml"))
    
    for file_path in rule_files:
        try:
            with open(file_path, 'r') as f:
                file_content = yaml.safe_load(f)
            
            # Skip empty files
            if not file_content:
                continue
            
            # Extract rules from each group
            for group in file_content.get('groups', []):
                for rule in group.get('rules', []):
                    # Skip recording rules (without alert field)
                    if 'alert' not in rule:
                        continue
                    
                    all_rules.append(AlertRule(
                        name=rule.get('alert', ''),
                        expr=rule.get('expr', ''),
                        file=os.path.basename(file_path),
                        for_duration=rule.get('for', ''),
                        labels=rule.get('labels', {}),
                        annotations=rule.get('annotations', {})
                    ))
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
    
    logger.info(f"Parsed {len(all_rules)} alert rules from {len(rule_files)} files")
    return all_rules

def find_overlapping_rules(rules: List[AlertRule], threshold: float = 0.8) -> List[Tuple[AlertRule, AlertRule, float]]:
    """Find pairs of rules with similarity above the threshold"""
    overlaps = []
    
    for i, rule1 in enumerate(rules):
        for j, rule2 in enumerate(rules[i+1:], i+1):
            # Skip rules with same name (likely in different files)
            if rule1.name == rule2.name:
                continue
            
            similarity = rule1.similarity_score(rule2)
            if similarity >= threshold:
                overlaps.append((rule1, rule2, similarity))
    
    # Sort by similarity (descending)
    overlaps.sort(key=lambda x: x[2], reverse=True)
    
    return overlaps

def generate_report(overlaps: List[Tuple[AlertRule, AlertRule, float]]) -> str:
    """Generate a markdown report of overlapping rules"""
    if not overlaps:
        return "# Alert Overlap Analysis\n\nNo significant overlaps detected."
    
    report = [
        "# Alert Overlap Analysis",
        "",
        f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Found {len(overlaps)} potential overlaps.",
        "",
        "## Overlapping Rules"
    ]
    
    for i, (rule1, rule2, similarity) in enumerate(overlaps, 1):
        overlap_pct = f"{similarity * 100:.1f}%"
        report.extend([
            f"### Overlap {i}: {overlap_pct} Similarity",
            "",
            "#### Rule 1",
            rule1.details(),
            "",
            "#### Rule 2",
            rule2.details(),
            "",
            "#### Comparison",
            "```diff",
        ])
        
        # Generate a diff of the expressions
        diff = difflib.ndiff(
            rule1.normalized_expr.splitlines(),
            rule2.normalized_expr.splitlines()
        )
        report.extend(diff)
        
        report.extend([
            "```",
            "",
            "---",
            ""
        ])
    
    report.extend([
        "## Recommendation",
        "",
        "Review the above overlaps and consider:",
        "",
        "1. **Merging rules** if they're intentionally similar but just split across files",
        "2. **Refining expressions** to make them more distinct if they're capturing different conditions",
        "3. **Using different time windows** for different severity levels of the same condition",
        "4. **Documenting exceptions** to explain why similar rules exist",
        "",
        "Remember that some overlap may be intentional for different warning levels."
    ])
    
    return "\n".join(report)

def send_report_email(report: str, email: str) -> bool:
    """Send report by email (placeholder - implement with your email system)"""
    try:
        logger.info(f"Would send email to {email}")
        # Add your email sending implementation here
        return True
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Prometheus Alert Overlap Checker")
    parser.add_argument('--rules-dir', type=str, default="prometheus/rules",
                        help='Directory containing Prometheus rule files')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Similarity threshold (0.0-1.0) for reporting overlaps')
    parser.add_argument('--report-file', type=str, default="alert_overlaps.md",
                        help='Output report filename')
    parser.add_argument('--email', type=str, help='Email address to send report to')
    args = parser.parse_args()
    
    # Parse rules
    rules = parse_rule_files(args.rules_dir)
    
    if not rules:
        logger.warning(f"No alert rules found in {args.rules_dir}")
        return
    
    # Find overlaps
    overlaps = find_overlapping_rules(rules, args.threshold)
    
    # Generate report
    report = generate_report(overlaps)
    
    # Write report to file
    with open(args.report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report written to {args.report_file}")
    
    # Send email if requested
    if args.email:
        send_report_email(report, args.email)

if __name__ == "__main__":
    main()
