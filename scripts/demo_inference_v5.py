#!/usr/bin/env python3
"""
Demo Inference Script v5
=========================
Interactive CLI for testing inference, adapters, mesh, and morphing.
Includes colored terminal output and performance benchmarking.
"""

import argparse
import json
import time
import asyncio
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import random

# Colored output support
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    print("[WARNING] colorama not installed. Install with: pip install colorama")
    
    # Fallback
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""

# Import our modules
sys.path.append(str(Path(__file__).parent.parent))

from python.core.saigon_inference_v5 import SaigonInference
from python.core.adapter_loader_v5 import MetadataManager
from python.core.concept_mesh_v5 import MeshManager
from python.core.conversation_manager import ConversationManager
from python.core.lattice_morphing import LatticeMorpher

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:8001"
DEMO_PROMPTS = [
    "Explain quantum computing",
    "What is a kagome lattice?",
    "How do solitons work in memory systems?",
    "Describe homotopy type theory",
    "What is a psi-morphon?",
    "Explain holographic rendering",
    "How does LoRA fine-tuning work?",
    "What is mesh context injection?"
]

TEST_USERS = ["alice", "bob", "charlie", "diana", "eve"]

# ============================================================================
# DEMO MODES
# ============================================================================

class DemoMode:
    """Base class for demo modes."""
    
    def __init__(self, api_url: str = API_BASE_URL):
        self.api_url = api_url
        self.inference_engine = SaigonInference()
        self.metadata_manager = MetadataManager()
        self.mesh_manager = MeshManager()
        
    def print_header(self, title: str):
        """Print colored header."""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{title.center(60)}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    def print_result(self, label: str, value: Any, color=Fore.WHITE):
        """Print colored result."""
        print(f"{Fore.YELLOW}{label}: {color}{value}{Style.RESET_ALL}")
    
    def print_error(self, message: str):
        """Print error message."""
        print(f"{Fore.RED}[ERROR] {message}{Style.RESET_ALL}")
    
    def print_success(self, message: str):
        """Print success message."""
        print(f"{Fore.GREEN}[SUCCESS] {message}{Style.RESET_ALL}")
    
    def print_info(self, message: str):
        """Print info message."""
        print(f"{Fore.BLUE}[INFO] {message}{Style.RESET_ALL}")

class InteractiveMode(DemoMode):
    """Interactive chat mode."""
    
    def run(self, user_id: str = "demo_user"):
        self.print_header("INTERACTIVE INFERENCE MODE")
        
        # Create conversation
        conversation_manager = ConversationManager(self.inference_engine, self.mesh_manager)
        conversation_id = conversation_manager.start_conversation(user_id)
        
        self.print_info(f"Started conversation: {conversation_id}")
        self.print_info("Type 'quit' to exit, 'help' for commands\n")
        
        while True:
            try:
                # Get user input
                user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                elif not user_input:
                    continue
                
                # Process message
                start_time = time.time()
                result = conversation_manager.process_message(
                    conversation_id=conversation_id,
                    user_input=user_input
                )
                latency = (time.time() - start_time) * 1000
                
                # Display response
                print(f"{Fore.CYAN}Assistant: {Fore.WHITE}{result['output']}")
                print(f"{Fore.YELLOW}[Latency: {latency:.2f}ms]{Style.RESET_ALL}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.print_error(str(e))
        
        # End conversation
        conversation_manager.end_conversation(conversation_id)
        self.print_info("Conversation ended")
    
    def show_help(self):
        """Show help commands."""
        print(f"\n{Fore.YELLOW}Commands:")
        print(f"  quit  - Exit interactive mode")
        print(f"  help  - Show this help")
        print(f"  stats - Show inference statistics\n")
    
    def show_stats(self):
        """Show inference statistics."""
        stats = self.inference_engine.get_statistics()
        print(f"\n{Fore.YELLOW}Statistics:")
        print(f"  Device: {stats['device']}")
        print(f"  Cache Size: {stats['cache']['size']}/{stats['cache']['maxsize']}")
        print(f"  Cache Hit Rate: {stats['cache']['hit_rate']:.2%}")
        print(f"  Total Requests: {stats['cache']['total_requests']}\n")

class PerformanceMode(DemoMode):
    """Performance benchmarking mode."""
    
    def run(self, user_id: str = "benchmark_user", iterations: int = 10):
        self.print_header("PERFORMANCE BENCHMARK MODE")
        
        results = {
            "cold_start": [],
            "cached": [],
            "with_mesh": [],
            "without_mesh": []
        }
        
        # Test prompts
        prompts = random.sample(DEMO_PROMPTS, min(iterations, len(DEMO_PROMPTS)))
        
        self.print_info(f"Running {iterations} iterations...")
        
        for i, prompt in enumerate(prompts):
            print(f"\n{Fore.YELLOW}Iteration {i+1}/{iterations}: {prompt[:50]}...")
            
            # Cold start
            self.inference_engine.clear_cache()
            start = time.time()
            result = self.inference_engine.run_inference(
                user_id=user_id,
                user_input=prompt,
                use_mesh_context=False
            )
            cold_latency = (time.time() - start) * 1000
            results["cold_start"].append(cold_latency)
            
            # Cached
            start = time.time()
            result = self.inference_engine.run_inference(
                user_id=user_id,
                user_input=prompt,
                use_mesh_context=False
            )
            cached_latency = (time.time() - start) * 1000
            results["cached"].append(cached_latency)
            
            # With mesh
            start = time.time()
            result = self.inference_engine.run_inference(
                user_id=user_id,
                user_input=prompt,
                use_mesh_context=True
            )
            mesh_latency = (time.time() - start) * 1000
            results["with_mesh"].append(mesh_latency)
            
            print(f"  Cold Start: {Fore.RED}{cold_latency:.2f}ms")
            print(f"  Cached: {Fore.GREEN}{cached_latency:.2f}ms")
            print(f"  With Mesh: {Fore.BLUE}{mesh_latency:.2f}ms")
        
        # Display summary
        self.print_header("BENCHMARK RESULTS")
        
        for category, latencies in results.items():
            if latencies:
                avg = sum(latencies) / len(latencies)
                min_lat = min(latencies)
                max_lat = max(latencies)
                
                print(f"\n{Fore.YELLOW}{category.replace('_', ' ').title()}:")
                print(f"  Average: {Fore.WHITE}{avg:.2f}ms")
                print(f"  Min: {Fore.GREEN}{min_lat:.2f}ms")
                print(f"  Max: {Fore.RED}{max_lat:.2f}ms")
        
        # Calculate speedup
        if results["cold_start"] and results["cached"]:
            speedup = sum(results["cold_start"]) / sum(results["cached"])
            print(f"\n{Fore.CYAN}Cache Speedup: {speedup:.2f}x")

class MultiUserMode(DemoMode):
    """Multi-user simulation mode."""
    
    def run(self, num_users: int = 5):
        self.print_header("MULTI-USER SIMULATION MODE")
        
        users = TEST_USERS[:num_users]
        self.print_info(f"Simulating {num_users} users: {', '.join(users)}")
        
        # Create sessions for each user
        sessions = {}
        for user in users:
            session_id = self._create_session(user)
            sessions[user] = session_id
            self.print_success(f"Created session for {user}: {session_id[:8]}...")
        
        # Simulate concurrent requests
        print(f"\n{Fore.YELLOW}Simulating concurrent requests...")
        
        for round in range(3):
            print(f"\n{Fore.CYAN}Round {round + 1}:")
            
            for user in users:
                prompt = random.choice(DEMO_PROMPTS)
                
                start = time.time()
                result = self.inference_engine.run_inference(
                    user_id=user,
                    user_input=prompt,
                    max_new_tokens=50
                )
                latency = (time.time() - start) * 1000
                
                adapter = result.get("adapter_used", "none")
                output = result["output"][:50] + "..." if len(result["output"]) > 50 else result["output"]
                
                print(f"  {Fore.GREEN}{user}: {Fore.WHITE}{output}")
                print(f"    {Fore.YELLOW}[{latency:.2f}ms, adapter: {adapter}]")
        
        # Show cache statistics
        stats = self.inference_engine.get_statistics()
        print(f"\n{Fore.CYAN}Final Statistics:")
        print(f"  Cache Hit Rate: {stats['cache']['hit_rate']:.2%}")
        print(f"  Cache Size: {stats['cache']['size']}/{stats['cache']['maxsize']}")
    
    def _create_session(self, user_id: str) -> str:
        """Create session via API."""
        try:
            response = requests.post(
                f"{self.api_url}/api/saigon/conversation/start",
                params={"user_id": user_id}
            )
            return response.json().get("conversation_id", "unknown")
        except:
            return f"local_{user_id}_{int(time.time())}"

class AdapterTestMode(DemoMode):
    """Adapter management test mode."""
    
    def run(self, user_id: str = "adapter_test"):
        self.print_header("ADAPTER MANAGEMENT TEST MODE")
        
        # List adapters
        adapters = self.metadata_manager.list_user_adapters(user_id)
        
        if not adapters:
            self.print_info(f"No adapters found for {user_id}")
            self.print_info("Creating sample adapter...")
            
            # This would normally train an adapter
            # For demo, we'll just show the flow
            self.print_success("Sample adapter created (simulated)")
            return
        
        print(f"{Fore.YELLOW}Found {len(adapters)} adapters for {user_id}:")
        for adapter in adapters:
            status = "[ACTIVE]" if adapter.get("active") else ""
            print(f"  - {adapter['adapter_id']} {Fore.GREEN}{status}")
            print(f"    Score: {adapter.get('score', 'N/A')}")
            print(f"    Created: {adapter['created']}")
        
        # Test hot-swapping
        if len(adapters) > 1:
            print(f"\n{Fore.YELLOW}Testing hot-swap...")
            
            inactive = [a for a in adapters if not a.get("active")]
            if inactive:
                target = inactive[0]
                
                self.print_info(f"Swapping to {target['adapter_id']}")
                success = self.metadata_manager.promote_adapter(user_id, target['adapter_id'])
                
                if success:
                    self.print_success("Hot-swap successful!")
                else:
                    self.print_error("Hot-swap failed")

class MorphingMode(DemoMode):
    """Lattice morphing demo mode."""
    
    def run(self):
        self.print_header("LATTICE MORPHING DEMO MODE")
        
        morpher = LatticeMorpher()
        targets = ["kagome", "cubic", "soliton"]
        
        for target in targets:
            self.print_info(f"Morphing to {target}...")
            
            # Start morphing
            morpher.morph_to(target, duration_ms=1000)
            
            # Show progress
            while morpher.morphing:
                progress = morpher.get_morph_progress()
                bar_length = 40
                filled = int(bar_length * progress)
                bar = "█" * filled + "░" * (bar_length - filled)
                
                print(f"\r  {Fore.CYAN}[{bar}] {progress:.0%}", end="")
                time.sleep(0.05)
            
            print()  # New line
            
            # Get final state
            state = morpher.get_current_state()
            if state:
                self.print_success(f"Morphed to {target}")
                print(f"  Phase: {state.phase:.3f}")
                print(f"  Coherence: {state.coherence:.3f}")
                print(f"  Energy: {state.energy:.3f}")
            
            time.sleep(0.5)

# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(description="TORI/Saigon Demo Inference")
    parser.add_argument(
        "--mode",
        choices=["interactive", "performance", "multiuser", "adapter", "morphing"],
        default="interactive",
        help="Demo mode to run"
    )
    parser.add_argument("--user_id", default="demo_user", help="User ID")
    parser.add_argument("--iterations", type=int, default=10, help="Performance test iterations")
    parser.add_argument("--num_users", type=int, default=5, help="Number of users for multiuser test")
    parser.add_argument("--api_url", default=API_BASE_URL, help="API base URL")
    
    args = parser.parse_args()
    
    # ASCII Art Banner
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║  {Fore.WHITE}{Style.BRIGHT}████████╗ ██████╗ ██████╗ ██╗    ██╗    ██████╗{Fore.CYAN}      ║
║  {Fore.WHITE}{Style.BRIGHT}╚══██╔══╝██╔═══██╗██╔══██╗██║    ██║   ██╔════╝{Fore.CYAN}      ║
║  {Fore.WHITE}{Style.BRIGHT}   ██║   ██║   ██║██████╔╝██║    ██║   ███████╗{Fore.CYAN}      ║
║  {Fore.WHITE}{Style.BRIGHT}   ██║   ██║   ██║██╔══██╗██║    ╚██╗ ██╔╝╚════╝{Fore.CYAN}     ║
║  {Fore.WHITE}{Style.BRIGHT}   ██║   ╚██████╔╝██║  ██║██║     ╚████╔╝{Fore.CYAN}             ║
║  {Fore.WHITE}{Style.BRIGHT}   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝      ╚═══╝{Fore.CYAN}              ║
║                                                          ║
║  {Fore.YELLOW}Saigon Inference Engine v5 - Production Ready{Fore.CYAN}          ║
║  {Fore.GREEN}Multi-User | Adapters | Mesh | Morphing{Fore.CYAN}                ║
╚══════════════════════════════════════════════════════════╝
{Style.RESET_ALL}
    """
    
    print(banner)
    
    # Initialize demo mode
    modes = {
        "interactive": InteractiveMode,
        "performance": PerformanceMode,
        "multiuser": MultiUserMode,
        "adapter": AdapterTestMode,
        "morphing": MorphingMode
    }
    
    mode_class = modes.get(args.mode, InteractiveMode)
    demo = mode_class(args.api_url)
    
    # Run demo
    try:
        if args.mode == "interactive":
            demo.run(args.user_id)
        elif args.mode == "performance":
            demo.run(args.user_id, args.iterations)
        elif args.mode == "multiuser":
            demo.run(args.num_users)
        elif args.mode == "adapter":
            demo.run(args.user_id)
        elif args.mode == "morphing":
            demo.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Demo interrupted by user")
    except Exception as e:
        print(f"\n{Fore.RED}Demo error: {e}")
    
    print(f"\n{Fore.CYAN}Demo completed. Thank you for using TORI/Saigon!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
