#!/usr/bin/env python
"""
Test script for the ψ-Sync Stability Monitoring System.

This script demonstrates how to run the various components of the
ψ-Sync system, including basic functionality, Koopman integration,
and ALAN bridge tests.

Usage:
    python run_psi_sync_tests.py [test_name]

Where test_name can be:
    - basic: Run the basic PsiSyncMonitor demo
    - koopman: Run the Koopman integration demo
    - bridge: Run the ALAN bridge test
    - all: Run all tests (default)
"""

import os
import sys
import argparse
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("psi_sync_tests")

def run_basic_test():
    """Run the basic PsiSyncMonitor demo."""
    print("\n" + "="*80)
    print("Running Basic ψ-Sync Monitor Demo")
    print("="*80)
    
    # Import the module
    from psi_sync_demo import run_test_scenario
    
    # Run the test
    run_test_scenario()
    
    print("\nBasic test completed.\n")

def run_koopman_test():
    """Run the Koopman integration demo."""
    print("\n" + "="*80)
    print("Running ψ-Sync + Koopman Integration Demo")
    print("="*80)
    
    # Import the module
    from psi_koopman_integration import run_demo
    
    # Run the test
    run_demo()
    
    print("\nKoopman integration test completed.\n")

def run_bridge_test():
    """Run the ALAN bridge test."""
    print("\n" + "="*80)
    print("Running ALAN ψ-Sync Bridge Demo")
    print("="*80)
    
    # Import the module and run the main example
    import alan_psi_sync_bridge
    
    # The module has a main section that runs automatically when imported
    # If it didn't run, we could call a function here
    
    print("\nALAN bridge test completed.\n")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run ψ-Sync stability system tests")
    parser.add_argument(
        "test", 
        nargs="?", 
        choices=["basic", "koopman", "bridge", "all"], 
        default="all",
        help="Test to run (default: all)"
    )
    args = parser.parse_args()
    
    # Run the selected test(s)
    if args.test == "basic" or args.test == "all":
        run_basic_test()
        
    if args.test == "koopman" or args.test == "all":
        run_koopman_test()
        
    if args.test == "bridge" or args.test == "all":
        run_bridge_test()

if __name__ == "__main__":
    main()
