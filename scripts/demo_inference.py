#!/usr/bin/env python3
"""
Demo Script for Saigon Inference System
========================================
Interactive demonstration of adapter-aware inference with hot-swapping.
"""

import json
import time
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from python.core.saigon_inference import SaigonInference
from python.core.adapter_loader import (
    AdapterManager,
    get_adapter_path_for_user,
    list_user_adapters
)

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")

def print_info(label: str, value: str):
    """Print colored info line."""
    print(f"{Colors.CYAN}{label}:{Colors.END} {value}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.END}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.END}")

def create_demo_mesh_context(user_id: str):
    """Create a demo mesh context for testing."""
    mesh_dir = Path("data/mesh_contexts")
    mesh_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_context = {
        "user_id": user_id,
        "summary": f"User {user_id} is interested in quantum computing, kagome lattices, and soliton memory systems.",
        "relevant_nodes": [
            {"id": "node_1", "label": "Kagome Lattice", "confidence": 0.95},
            {"id": "node_2", "label": "Soliton Memory", "confidence": 0.88},
            {"id": "node_3", "label": "Quantum Computing", "confidence": 0.92}
        ],
        "edges": [
            {"source": "node_1", "target": "node_2", "relationship": "enables"},
            {"source": "node_2", "target": "node_3", "relationship": "implements"}
        ],
        "last_updated": time.time()
    }
    
    mesh_path = mesh_dir / f"user_{user_id}_mesh.json"
    with open(mesh_path, 'w') as f:
        json.dump(mesh_context, f, indent=2)
    
    return mesh_path

def create_demo_adapter(user_id: str):
    """Create a demo adapter file for testing."""
    adapters_dir = Path("models/adapters")
    adapters_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy adapter file (in real scenario, this would be actual LoRA weights)
    adapter_path = adapters_dir / f"user_{user_id}_lora.pt"
    
    # Create dummy weights (just for demo)
    import torch
    dummy_weights = {
        "lora_A": torch.randn(768, 16),
        "lora_B": torch.randn(16, 768),
        "config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"]
        }
    }
    torch.save(dummy_weights, adapter_path)
    
    return adapter_path

def demo_basic_inference():
    """Demonstrate basic inference."""
    print_header("BASIC INFERENCE DEMO")
    
    # Initialize engine
    print("\nInitializing inference engine...")
    engine = SaigonInference(
        base_model_dir="models/saigon_base/",
        use_cache=True
    )
    print_success("Engine initialized")
    
    # Create demo data
    user_id = "demo_user"
    print(f"\nCreating demo data for user: {user_id}")
    
    mesh_path = create_demo_mesh_context(user_id)
    print_success(f"Created mesh context: {mesh_path}")
    
    adapter_path = create_demo_adapter(user_id)
    print_success(f"Created adapter: {adapter_path}")
    
    # Run inference
    print("\n" + "-"*60)
    test_prompts = [
        "What is a kagome lattice?",
        "How does soliton memory work?",
        "Explain quantum computing basics."
    ]
    
    for prompt in test_prompts:
        print(f"\n{Colors.BLUE}Input:{Colors.END} {prompt}")
        
        try:
            result = engine.run_inference(
                user_id=user_id,
                user_input=prompt,
                use_mesh_context=True,
                max_length=128,
                temperature=0.7
            )
            
            print_info("Adapter", result.get("adapter_used", "None"))
            print_info("Mesh Context", "Yes" if result["mesh_context_used"] else "No")
            print_info("Latency", f"{result['latency_ms']:.2f}ms")
            print(f"{Colors.GREEN}Output:{Colors.END} {result['output'][:200]}...")
            
        except Exception as e:
            print_error(f"Inference failed: {e}")

def demo_hot_swapping():
    """Demonstrate hot-swapping adapters."""
    print_header("HOT-SWAPPING DEMO")
    
    # Create multiple adapters
    print("\nCreating multiple adapter versions...")
    adapters_dir = Path("models/adapters")
    adapters_dir.mkdir(parents=True, exist_ok=True)
    
    # Create v1 and v2 adapters
    import torch
    for version in ["v1", "v2"]:
        adapter_path = adapters_dir / f"demo_adapter_{version}.pt"
        dummy_weights = {
            "version": version,
            "lora_A": torch.randn(768, 16),
            "lora_B": torch.randn(16, 768)
        }
        torch.save(dummy_weights, adapter_path)
        print_success(f"Created adapter: {adapter_path}")
    
    # Initialize manager
    manager = AdapterManager(adapters_dir)
    
    # Test hot-swapping
    print("\n" + "-"*60)
    print("Testing hot-swap functionality...")
    
    v1_path = str(adapters_dir / "demo_adapter_v1.pt")
    v2_path = str(adapters_dir / "demo_adapter_v2.pt")
    
    success = manager.hot_swap_adapter(v1_path, v2_path)
    if success:
        print_success("Hot-swap successful!")
    else:
        print_error("Hot-swap failed")

def demo_adapter_management():
    """Demonstrate adapter management features."""
    print_header("ADAPTER MANAGEMENT DEMO")
    
    # Initialize manager
    manager = AdapterManager()
    
    # List available adapters
    print("\nAvailable adapters:")
    adapters = manager.list_available_adapters()
    
    if adapters:
        for adapter in adapters:
            print(f"  • {Path(adapter['path']).name}")
            print(f"    Size: {adapter['size_bytes'] / 1024:.2f} KB")
            print(f"    Modified: {adapter['modified_time']}")
    else:
        print_warning("No adapters found")
    
    # Test user adapter listing
    print("\n" + "-"*60)
    user_id = "demo_user"
    user_adapters = list_user_adapters(user_id)
    
    if user_adapters:
        print(f"Adapters for user '{user_id}':")
        for adapter in user_adapters:
            print(f"  • {adapter}")
    else:
        print_warning(f"No adapters found for user '{user_id}'")

def demo_performance_test():
    """Demonstrate performance testing."""
    print_header("PERFORMANCE TEST")
    
    engine = SaigonInference(use_cache=True)
    user_id = "perf_test"
    
    # Create test data
    create_demo_mesh_context(user_id)
    create_demo_adapter(user_id)
    
    # Test with and without cache
    test_prompt = "Explain the concept of quantum entanglement."
    
    print("\nTesting inference performance...")
    print(f"Prompt: {test_prompt}")
    print("-"*60)
    
    # First run (cache miss)
    start = time.time()
    result1 = engine.run_inference(
        user_id=user_id,
        user_input=test_prompt,
        max_length=50
    )
    time1 = (time.time() - start) * 1000
    
    # Second run (cache hit)
    start = time.time()
    result2 = engine.run_inference(
        user_id=user_id,
        user_input=test_prompt,
        max_length=50
    )
    time2 = (time.time() - start) * 1000
    
    print_info("First run (cache miss)", f"{time1:.2f}ms")
    print_info("Second run (cache hit)", f"{time2:.2f}ms")
    print_info("Speedup", f"{time1/time2:.2f}x")
    
    # Clear cache
    print("\n" + "-"*60)
    print("Clearing cache...")
    engine.clear_cache()
    print_success("Cache cleared")

def interactive_demo():
    """Interactive demo mode."""
    print_header("INTERACTIVE INFERENCE DEMO")
    
    # Initialize engine
    engine = SaigonInference(use_cache=True)
    
    # Get user ID
    user_id = input(f"\n{Colors.CYAN}Enter user ID (or 'demo' for demo user):{Colors.END} ").strip()
    if not user_id or user_id == "demo":
        user_id = "demo_user"
        create_demo_mesh_context(user_id)
        create_demo_adapter(user_id)
    
    print(f"\n{Colors.GREEN}Using user: {user_id}{Colors.END}")
    print("Type 'quit' to exit, 'clear' to clear cache, 'status' for engine status")
    print("-"*60)
    
    while True:
        try:
            # Get user input
            prompt = input(f"\n{Colors.BLUE}You:{Colors.END} ").strip()
            
            if prompt.lower() == 'quit':
                print("\nGoodbye!")
                break
            
            if prompt.lower() == 'clear':
                engine.clear_cache()
                print_success("Cache cleared")
                continue
            
            if prompt.lower() == 'status':
                print_info("Current User", engine.current_user_id or "None")
                print_info("Current Adapter", engine.current_adapter_path or "None")
                print_info("Device", engine.device)
                print_info("Cache Enabled", str(engine.use_cache))
                continue
            
            if not prompt:
                continue
            
            # Run inference
            start = time.time()
            result = engine.run_inference(
                user_id=user_id,
                user_input=prompt,
                use_mesh_context=True,
                max_length=256,
                temperature=0.7
            )
            
            # Display result
            print(f"\n{Colors.GREEN}Saigon:{Colors.END} {result['output']}")
            print(f"\n{Colors.CYAN}[Latency: {result['latency_ms']:.2f}ms | "
                  f"Adapter: {Path(result['adapter_used']).name if result['adapter_used'] else 'None'} | "
                  f"Mesh: {'Yes' if result['mesh_context_used'] else 'No'}]{Colors.END}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print_error(f"Error: {e}")

def main():
    """Main demo menu."""
    print_header("SAIGON INFERENCE SYSTEM DEMO")
    print("\nWelcome to the Saigon Adapter-Aware Inference Demo!")
    print("This demonstrates the complete inference pipeline with:")
    print("  • LoRA adapter loading and hot-swapping")
    print("  • Mesh context injection")
    print("  • LRU caching for performance")
    print("  • Multi-user support")
    
    while True:
        print("\n" + "="*60)
        print("Select a demo:")
        print("  1. Basic Inference")
        print("  2. Hot-Swapping")
        print("  3. Adapter Management")
        print("  4. Performance Test")
        print("  5. Interactive Mode")
        print("  0. Exit")
        print("="*60)
        
        choice = input("\nEnter choice (0-5): ").strip()
        
        if choice == "0":
            print("\nGoodbye!")
            break
        elif choice == "1":
            demo_basic_inference()
        elif choice == "2":
            demo_hot_swapping()
        elif choice == "3":
            demo_adapter_management()
        elif choice == "4":
            demo_performance_test()
        elif choice == "5":
            interactive_demo()
        else:
            print_warning("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
