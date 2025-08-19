#!/usr/bin/env python3
"""
Automated Memory Vault Audit Runner
Can be scheduled as a cron job or Windows task
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from audit_memory_vault import audit_vault

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_scheduled_audit(generate_html: bool = False):
    """Run audit and save results with timestamp"""
    
    vault_path = Path("data/memory_vault/memories")
    audit_dir = Path("data/memory_vault/audits")
    audit_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"🔍 Running scheduled audit at {timestamp}")
    
    # Run the audit
    results = audit_vault(vault_path, fix=False)
    
    # Add timestamp to results
    audit_record = {
        "timestamp": timestamp.isoformat(),
        "date": timestamp.strftime("%Y-%m-%d"),
        "time": timestamp.strftime("%H:%M:%S"),
        "results": results,
        "vault_path": str(vault_path),
        "files_checked": results.get('total', 0),
        "files_passed": results.get('passed', 0),
        "files_failed": results.get('failed', 0),
        "health_score": (results.get('passed', 0) / results.get('total', 1)) * 100 if results.get('total', 0) > 0 else 0
    }
    
    # Save JSON log
    json_log_path = audit_dir / f"audit_{date_str}.json"
    with open(json_log_path, 'w', encoding='utf-8') as f:
        json.dump(audit_record, f, indent=2)
    
    logger.info(f"📄 Audit log saved to: {json_log_path}")
    
    # Append to cumulative log
    cumulative_log_path = audit_dir / "audit_history.jsonl"
    with open(cumulative_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(audit_record) + '\n')
    
    # Generate HTML report if requested
    if generate_html:
        html_path = audit_dir / f"audit_{date_str}.html"
        generate_html_report(audit_record, html_path)
        logger.info(f"📊 HTML report saved to: {html_path}")
    
    # Check if intervention needed
    if audit_record['health_score'] < 80:
        logger.warning(f"⚠️ Vault health score low: {audit_record['health_score']:.1f}%")
        logger.warning("   Consider running: poetry run python audit_memory_vault.py --fix")
    else:
        logger.info(f"✅ Vault health score: {audit_record['health_score']:.1f}%")
    
    return audit_record


def generate_html_report(audit_record: dict, output_path: Path):
    """Generate a nice HTML report from audit results"""
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Memory Vault Audit Report - {date}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: #2196F3;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .metric {{
            display: inline-block;
            background: white;
            padding: 15px;
            margin: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .health-score {{
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            background: {health_color};
            color: white;
            border-radius: 50%;
            width: 120px;
            height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px auto;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Memory Vault Audit Report</h1>
        <div class="timestamp">Generated: {timestamp}</div>
    </div>
    
    <div class="health-score">{health_score:.0f}%</div>
    
    <div style="text-align: center;">
        <div class="metric">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total Files</div>
        </div>
        <div class="metric">
            <div class="metric-value">{passed}</div>
            <div class="metric-label">Passed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{failed}</div>
            <div class="metric-label">Failed</div>
        </div>
    </div>
    
    <div style="margin-top: 40px; padding: 20px; background: white; border-radius: 8px;">
        <h3>Audit Details</h3>
        <p><strong>Vault Path:</strong> {vault_path}</p>
        <p><strong>Date:</strong> {date}</p>
        <p><strong>Time:</strong> {time}</p>
        <p><strong>Health Score:</strong> {health_score:.1f}%</p>
    </div>
</body>
</html>
    """
    
    # Determine health color
    health_score = audit_record['health_score']
    if health_score >= 90:
        health_color = '#4CAF50'  # Green
    elif health_score >= 70:
        health_color = '#FF9800'  # Orange
    else:
        health_color = '#F44336'  # Red
    
    # Fill template
    html_content = html_template.format(
        date=audit_record['date'],
        timestamp=audit_record['timestamp'],
        health_score=health_score,
        health_color=health_color,
        total=audit_record['files_checked'],
        passed=audit_record['files_passed'],
        failed=audit_record['files_failed'],
        vault_path=audit_record['vault_path'],
        time=audit_record['time']
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def show_audit_history():
    """Display audit history summary"""
    history_path = Path("data/memory_vault/audits/audit_history.jsonl")
    
    if not history_path.exists():
        logger.info("No audit history found")
        return
    
    records = []
    with open(history_path, 'r') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    
    logger.info(f"\n📊 Audit History ({len(records)} audits)")
    logger.info("="*60)
    
    for record in records[-10:]:  # Show last 10
        logger.info(f"{record['date']} {record['time']} - "
                   f"Health: {record['health_score']:.1f}% "
                   f"({record['files_passed']}/{record['files_checked']} passed)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run scheduled memory vault audit")
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--history', action='store_true', help='Show audit history')
    
    args = parser.parse_args()
    
    if args.history:
        show_audit_history()
    else:
        run_scheduled_audit(generate_html=args.html)
