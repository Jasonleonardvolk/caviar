#!/usr/bin/env python3
"""
Runner script for the ELFIN Manipulator Controller Analyzer.
This script analyzes the controller for correctness and generates simulation code.
"""

import os
import sys
from pathlib import Path
import json

# Import the analyzer from analyze_manipulator.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from analyze_manipulator import ELFINAnalyzer, prepare_simulation_scaffold
except ImportError:
    print("Error: Could not import analyzer. Make sure analyze_manipulator.py is in the same directory.")
    sys.exit(1)

def main():
    # Define paths
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    controller_path = current_dir / "src" / "manipulator_controller.elfin"
    output_dir = current_dir / "output"
    results_path = output_dir / "analysis_results.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and run analyzer
    print(f"Analyzing manipulator controller: {controller_path}")
    analyzer = ELFINAnalyzer(controller_path)
    analyzer.run_analysis()
    
    # Print summary and export results
    analyzer.print_summary()
    analyzer.export_results(results_path)
    
    # Generate simulation scaffolding
    print("\nGenerating simulation scaffold...")
    prepare_simulation_scaffold(analyzer, output_dir)
    print(f"Simulation scaffold generated in: {output_dir}")
    
    # Validate absence of circular references
    error_count = sum(1 for issue in analyzer.issues if issue['severity'] == 'error')
    if error_count == 0:
        print("\n✅ No circular references or mathematical errors found in the ELFIN specification!")
        print("The file is ready for formal verification and simulation.")
    else:
        print(f"\n❌ Found {error_count} errors in the ELFIN specification.")
        print("Please fix these issues before proceeding with simulation or verification.")
        
    # Summarize the sections and elements found
    section_counts = {}
    for section_key in analyzer.sections:
        section_type = section_key.split(':')[0]
        if section_type not in section_counts:
            section_counts[section_type] = 0
        section_counts[section_type] += 1
        
    print("\nELFIN Specification Component Summary:")
    print(f"- System:    {section_counts.get('system', 0)}")
    print(f"- Lyapunov:  {section_counts.get('lyapunov', 0)}")
    print(f"- Barriers:  {section_counts.get('barrier', 0)}")
    print(f"- Controllers: {section_counts.get('mode', 0)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
