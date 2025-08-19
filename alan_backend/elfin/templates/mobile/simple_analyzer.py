#!/usr/bin/env python3
"""
Simple ELFIN syntax checker for mobile_robot_controller.elfin
"""

import os
import sys
from pathlib import Path

# Path to the ELFIN file
elfin_path = Path('alan_backend/elfin/templates/mobile/src/mobile_robot_controller.elfin')

# Read the file
with open(elfin_path, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File loaded: {len(content)} bytes")

# Check for specific syntax issues
issues = []

# Check for helpers: { vs helpers {
if "helpers: {" in content:
    issues.append("Invalid syntax: 'helpers: {' should be 'helpers {'")
else:
    print("✅ Correct helpers syntax found")

# Check for system: reference vs system reference
if "system: " in content:
    issues.append("Invalid syntax: 'system: ' reference should be direct system name")
else:
    print("✅ Correct system reference syntax")

# Check for flow_dynamics section existence
if "flow_dynamics {" in content:
    print("✅ flow_dynamics section found")
else:
    issues.append("Missing flow_dynamics section in system")

# Check for lyapunov: vs lyapunov
if "lyapunov: " in content:
    issues.append("Invalid syntax: 'lyapunov: ' should be 'lyapunov '")
else:
    print("✅ Correct lyapunov syntax")

# Check for barriers: vs barriers
if "barriers: [" in content:
    issues.append("Invalid syntax: 'barriers: [' should be 'barriers ['")
else:
    print("✅ Correct barriers syntax")

# Check for params: vs params
if "params: {" in content:
    issues.append("Invalid syntax: 'params: {' should be 'params {'")
else:
    print("✅ Correct params syntax")

# Check for planner: vs planner 
if "planner: " in content:
    issues.append("Invalid syntax: 'planner: ' should be 'planner '")
else:
    print("✅ Correct planner syntax")

# Check for controller: vs controller
if "controller: " in content:
    issues.append("Invalid syntax: 'controller: ' should be 'controller '")
else:
    print("✅ Correct controller syntax")

# Output summary
print("\n=== ELFIN Syntax Check Summary ===")
if issues:
    print(f"Found {len(issues)} issues:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    print("\n❌ Syntax issues found. Please fix before proceeding.")
else:
    print("✅ No syntax issues found! The file appears to be correct.")
