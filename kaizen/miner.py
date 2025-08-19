#!/usr/bin/env python3
"""
Kaizen Gap Miner
Analyzes performance logs to identify improvement opportunities
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceGap:
    """Identified performance gap"""
    gap_type: str
    severity: float  # 0-1 scale
    metric: str
    current_value: float
    target_value: float
    timestamp: float
    recommendations: List[str]

def find_gaps(log_path: Path) -> List[PerformanceGap]:
    """
    Analyze introspection log to find performance gaps
    
    Args:
        log_path: Path to introspection_meso.jl file
        
    Returns:
        List of identified gaps
    """
    gaps = []
    
    try:
        # Read log entries
        entries = []
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
                    
        if not entries:
            logger.warning("No entries found in log")
            return gaps
            
        # Analyze different metrics
        gaps.extend(_analyze_efficiency_gaps(entries))
        gaps.extend(_analyze_stability_gaps(entries))
        gaps.extend(_analyze_resource_gaps(entries))
        
        # Sort by severity
        gaps.sort(key=lambda g: g.severity, reverse=True)
        
        logger.info(f"Found {len(gaps)} performance gaps")
        
    except Exception as e:
        logger.error(f"Gap analysis failed: {e}")
        
    return gaps

def _analyze_efficiency_gaps(entries: List[Dict[str, Any]]) -> List[PerformanceGap]:
    """Analyze chaos efficiency metrics"""
    gaps = []
    
    # Extract efficiency values
    efficiencies = []
    for entry in entries:
        if 'efficiency' in entry:
            efficiencies.append(entry['efficiency']['mean'])
            
    if not efficiencies:
        return gaps
        
    # Check for low efficiency
    mean_efficiency = np.mean(efficiencies)
    if mean_efficiency < 3.0:  # Target is 3x minimum
        gaps.append(PerformanceGap(
            gap_type="low_chaos_efficiency",
            severity=0.8,
            metric="energy_efficiency",
            current_value=mean_efficiency,
            target_value=3.0,
            timestamp=entries[-1]['timestamp'],
            recommendations=[
                "Increase chaos burst intensity",
                "Optimize soliton propagation parameters",
                "Review attractor hopping thresholds"
            ]
        ))
        
    # Check for efficiency variance
    if len(efficiencies) > 10:
        efficiency_std = np.std(efficiencies)
        if efficiency_std > 1.0:
            gaps.append(PerformanceGap(
                gap_type="unstable_efficiency",
                severity=0.6,
                metric="efficiency_variance",
                current_value=efficiency_std,
                target_value=0.5,
                timestamp=entries[-1]['timestamp'],
                recommendations=[
                    "Stabilize chaos parameters",
                    "Implement adaptive control",
                    "Review energy budget allocation"
                ]
            ))
            
    return gaps

def _analyze_stability_gaps(entries: List[Dict[str, Any]]) -> List[PerformanceGap]:
    """Analyze eigenvalue stability metrics"""
    gaps = []
    
    # Extract lambda_max values
    lambda_values = []
    for entry in entries:
        if 'lambda_max' in entry:
            lambda_values.append(entry['lambda_max']['max'])
            
    if not lambda_values:
        return gaps
        
    # Check for instability trends
    max_lambda = np.max(lambda_values)
    if max_lambda > 0.9:  # Getting close to instability
        gaps.append(PerformanceGap(
            gap_type="eigenvalue_instability",
            severity=0.9,
            metric="max_eigenvalue",
            current_value=max_lambda,
            target_value=0.7,
            timestamp=entries[-1]['timestamp'],
            recommendations=[
                "Increase damping in unstable modes",
                "Review coupling matrix parameters",
                "Enable emergency stabilization"
            ]
        ))
        
    # Check for instability growth
    if len(lambda_values) > 20:
        recent = lambda_values[-10:]
        older = lambda_values[-20:-10]
        if np.mean(recent) > np.mean(older) * 1.2:
            gaps.append(PerformanceGap(
                gap_type="growing_instability",
                severity=0.7,
                metric="eigenvalue_trend",
                current_value=np.mean(recent),
                target_value=np.mean(older),
                timestamp=entries[-1]['timestamp'],
                recommendations=[
                    "Investigate instability source",
                    "Reduce system coupling strength",
                    "Implement trend-based damping"
                ]
            ))
            
    return gaps

def _analyze_resource_gaps(entries: List[Dict[str, Any]]) -> List[PerformanceGap]:
    """Analyze resource utilization"""
    gaps = []
    
    # Extract resource metrics
    cpu_values = []
    memory_values = []
    
    for entry in entries:
        if 'cpu' in entry:
            cpu_values.append(entry['cpu']['mean'])
        if 'memory' in entry:
            memory_values.append(entry['memory']['mean'])
            
    # Check CPU utilization
    if cpu_values:
        mean_cpu = np.mean(cpu_values)
        if mean_cpu < 20:  # Underutilized
            gaps.append(PerformanceGap(
                gap_type="cpu_underutilization",
                severity=0.4,
                metric="cpu_percent",
                current_value=mean_cpu,
                target_value=50,
                timestamp=entries[-1]['timestamp'],
                recommendations=[
                    "Increase parallel processing",
                    "Enable more chaos modes",
                    "Reduce idle time"
                ]
            ))
        elif mean_cpu > 80:  # Overutilized
            gaps.append(PerformanceGap(
                gap_type="cpu_overutilization",
                severity=0.7,
                metric="cpu_percent",
                current_value=mean_cpu,
                target_value=60,
                timestamp=entries[-1]['timestamp'],
                recommendations=[
                    "Optimize computational algorithms",
                    "Implement load balancing",
                    "Consider distributed processing"
                ]
            ))
            
    # Check memory usage
    if memory_values:
        mean_memory = np.mean(memory_values)
        if mean_memory > 85:
            gaps.append(PerformanceGap(
                gap_type="high_memory_usage",
                severity=0.8,
                metric="memory_percent",
                current_value=mean_memory,
                target_value=70,
                timestamp=entries[-1]['timestamp'],
                recommendations=[
                    "Implement memory consolidation",
                    "Review cache policies",
                    "Enable memory-mapped operations"
                ]
            ))
            
    return gaps

def generate_improvement_plan(gaps: List[PerformanceGap]) -> Dict[str, Any]:
    """Generate actionable improvement plan from gaps"""
    
    # Group gaps by type
    gap_groups = {}
    for gap in gaps:
        if gap.gap_type not in gap_groups:
            gap_groups[gap.gap_type] = []
        gap_groups[gap.gap_type].append(gap)
        
    # Create prioritized action plan
    plan = {
        'generated_at': datetime.now().isoformat(),
        'total_gaps': len(gaps),
        'priority_actions': [],
        'research_topics': [],
        'monitoring_focus': []
    }
    
    # High priority actions (severity > 0.7)
    for gap in gaps:
        if gap.severity > 0.7:
            plan['priority_actions'].extend(gap.recommendations)
            
    # Research topics based on gap patterns
    if 'low_chaos_efficiency' in gap_groups:
        plan['research_topics'].append({
            'topic': 'Advanced chaos control algorithms',
            'keywords': ['edge-of-chaos', 'soliton dynamics', 'energy harvesting'],
            'priority': 'high'
        })
        
    if 'eigenvalue_instability' in gap_groups:
        plan['research_topics'].append({
            'topic': 'Stability analysis and control',
            'keywords': ['Lyapunov methods', 'adaptive damping', 'spectral analysis'],
            'priority': 'critical'
        })
        
    # Monitoring focus areas
    for gap_type, group in gap_groups.items():
        avg_severity = np.mean([g.severity for g in group])
        if avg_severity > 0.5:
            plan['monitoring_focus'].append({
                'metric': group[0].metric,
                'frequency': 'increased',
                'threshold': group[0].target_value
            })
            
    # Remove duplicates
    plan['priority_actions'] = list(set(plan['priority_actions']))
    
    return plan

# Test function
def test_gap_miner():
    """Test the gap miner"""
    print("⛏️ Testing Gap Miner")
    print("=" * 50)
    
    # Create test log entries
    test_entries = [
        {
            'timestamp': 1234567890,
            'efficiency': {'mean': 2.5, 'min': 1.8},
            'lambda_max': {'mean': 0.85, 'max': 0.92},
            'cpu': {'mean': 15},
            'memory': {'mean': 88}
        },
        {
            'timestamp': 1234567900,
            'efficiency': {'mean': 2.2, 'min': 1.5},
            'lambda_max': {'mean': 0.88, 'max': 0.95},
            'cpu': {'mean': 12},
            'memory': {'mean': 90}
        }
    ]
    
    # Write test log
    test_log = Path("test_introspection.jl")
    with open(test_log, 'w') as f:
        for entry in test_entries:
            f.write(json.dumps(entry) + '\n')
            
    # Find gaps
    gaps = find_gaps(test_log)
    
    print(f"\nFound {len(gaps)} gaps:")
    for gap in gaps:
        print(f"\n  Type: {gap.gap_type}")
        print(f"  Severity: {gap.severity:.2f}")
        print(f"  Current: {gap.current_value:.2f} → Target: {gap.target_value:.2f}")
        print(f"  Recommendations:")
        for rec in gap.recommendations:
            print(f"    - {rec}")
            
    # Generate plan
    plan = generate_improvement_plan(gaps)
    print(f"\nImprovement Plan:")
    print(f"  Priority actions: {len(plan['priority_actions'])}")
    print(f"  Research topics: {len(plan['research_topics'])}")
    
    # Cleanup
    test_log.unlink()

if __name__ == "__main__":
    test_gap_miner()
