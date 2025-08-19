#!/usr/bin/env python3
"""
TORI/Saigon System Setup Script
================================
Initial setup and configuration for the complete system.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import subprocess
import platform

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_banner():
    """Print setup banner."""
    banner = f"""
{CYAN}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║  {WHITE}{BOLD}TORI/Saigon System Setup{CYAN}                               ║
║  {YELLOW}Production-Ready Self-Improving AI{CYAN}                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{RESET}
    """
    print(banner)

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"{RED}✗ Python 3.10+ required. Found: {sys.version}{RESET}")
        return False
    print(f"{GREEN}✓ Python {version.major}.{version.minor} detected{RESET}")
    return True

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"{GREEN}✓ CUDA available: {torch.cuda.get_device_name(0)}{RESET}")
            return True
        else:
            print(f"{YELLOW}⚠ CUDA not available. Using CPU mode.{RESET}")
            return False
    except ImportError:
        print(f"{YELLOW}⚠ PyTorch not installed yet{RESET}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "models/saigon_base",
        "models/adapters/backups",
        "data/mesh_contexts/backups",
        "data/training",
        "data/validation",
        "data/intent_vault",
        "data/psi_archive",
        "logs/inference",
        "logs/validation",
        "logs/rollback",
        "logs/conversations",
        "logs/user_context",
        "monitoring",
        "nginx/ssl"
    ]
    
    print(f"\n{BLUE}Creating directories...{RESET}")
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  {GREEN}✓{RESET} {dir_path}")
    
    # Create .gitkeep files
    gitkeep_dirs = [
        "models/adapters",
        "data/mesh_contexts",
        "data/training",
        "data/validation",
        "data/intent_vault",
        "data/psi_archive",
        "logs/inference",
        "logs/validation"
    ]
    
    for dir_path in gitkeep_dirs:
        gitkeep = Path(dir_path) / ".gitkeep"
        gitkeep.touch()

def install_dependencies():
    """Install Python dependencies."""
    print(f"\n{BLUE}Installing Python dependencies...{RESET}")
    
    # Check if in virtual environment
    if sys.prefix == sys.base_prefix:
        print(f"{YELLOW}⚠ Not in virtual environment. Recommended to use venv.{RESET}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print(f"{GREEN}✓ Python dependencies installed{RESET}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{RED}✗ Failed to install dependencies: {e}{RESET}")
        return False

def initialize_metadata():
    """Initialize adapter metadata file."""
    metadata_file = Path("models/adapters/metadata.json")
    
    if not metadata_file.exists():
        print(f"\n{BLUE}Initializing adapter metadata...{RESET}")
        metadata = {
            "global": [{
                "adapter_id": "global_v1",
                "path": "models/adapters/global_adapter_v1.pt",
                "version": "1.0.0",
                "active": True,
                "created": "2025-01-01T00:00:00Z",
                "description": "Global default adapter"
            }]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"{GREEN}✓ Metadata initialized{RESET}")
    else:
        print(f"{GREEN}✓ Metadata already exists{RESET}")

def setup_frontend():
    """Setup frontend dependencies."""
    print(f"\n{BLUE}Setting up frontend...{RESET}")
    
    frontend_dir = Path("frontend/hybrid")
    if not frontend_dir.exists():
        print(f"{YELLOW}⚠ Frontend directory not found{RESET}")
        return False
    
    # Check if npm is available
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{YELLOW}⚠ npm not found. Please install Node.js{RESET}")
        return False
    
    # Install frontend dependencies
    try:
        os.chdir(frontend_dir)
        subprocess.run(["npm", "install"], check=True)
        os.chdir("../..")
        print(f"{GREEN}✓ Frontend dependencies installed{RESET}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{RED}✗ Failed to install frontend dependencies: {e}{RESET}")
        return False

def create_env_file():
    """Create .env file with default configuration."""
    env_file = Path(".env")
    
    if not env_file.exists():
        print(f"\n{BLUE}Creating .env file...{RESET}")
        
        env_content = """# TORI/Saigon Environment Configuration

# Core Settings
SAIGON_BASE_MODEL=models/saigon_base
SAIGON_ADAPTERS_DIR=models/adapters
SAIGON_MESH_DIR=data/mesh_contexts
SAIGON_DEVICE=cuda
SAIGON_CACHE_SIZE=5
SAIGON_LOG_LEVEL=INFO

# API Settings
API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=4

# Frontend
NODE_ENV=development
API_URL=http://localhost:8001

# Database (optional)
POSTGRES_DB=tori
POSTGRES_USER=tori_user
POSTGRES_PASSWORD=change_me_in_production
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=change_me_in_production_to_random_string
CORS_ORIGINS=http://localhost:3000,http://localhost:8001
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"{GREEN}✓ .env file created{RESET}")
        print(f"{YELLOW}  ⚠ Remember to update passwords and secrets for production!{RESET}")
    else:
        print(f"{GREEN}✓ .env file already exists{RESET}")

def download_base_model():
    """Download base model if needed."""
    print(f"\n{BLUE}Checking base model...{RESET}")
    
    base_model_dir = Path("models/saigon_base")
    
    # Check if model already exists
    if (base_model_dir / "config.json").exists():
        print(f"{GREEN}✓ Base model already exists{RESET}")
        return True
    
    print(f"{YELLOW}Base model not found. Would you like to download it?{RESET}")
    print("Options:")
    print("  1. microsoft/phi-2 (2.7B parameters, recommended)")
    print("  2. TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B, smaller)")
    print("  3. Skip (configure manually later)")
    
    choice = input("Choice (1/2/3): ")
    
    if choice == "3":
        print(f"{YELLOW}⚠ Skipping model download. Configure manually later.{RESET}")
        return False
    
    model_map = {
        "1": "microsoft/phi-2",
        "2": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    }
    
    model_name = model_map.get(choice, "microsoft/phi-2")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"{BLUE}Downloading {model_name}...{RESET}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"{BLUE}Saving to {base_model_dir}...{RESET}")
        model.save_pretrained(base_model_dir)
        tokenizer.save_pretrained(base_model_dir)
        
        print(f"{GREEN}✓ Base model downloaded successfully{RESET}")
        return True
        
    except Exception as e:
        print(f"{RED}✗ Failed to download model: {e}{RESET}")
        return False

def create_sample_data():
    """Create sample data for testing."""
    print(f"\n{BLUE}Creating sample data...{RESET}")
    
    # Sample mesh context
    sample_mesh = {
        "user_id": "demo_user",
        "summary": "Demo user interested in quantum computing and AI",
        "nodes": [
            {"id": "n1", "label": "Quantum Computing", "confidence": 0.9},
            {"id": "n2", "label": "Machine Learning", "confidence": 0.85},
            {"id": "n3", "label": "Neural Networks", "confidence": 0.8}
        ],
        "edges": [
            {"source": "n1", "target": "n2", "relationship": "enables"},
            {"source": "n2", "target": "n3", "relationship": "implements"}
        ],
        "version": "1.0.0"
    }
    
    mesh_file = Path("data/mesh_contexts/user_demo_user_mesh.json")
    with open(mesh_file, 'w') as f:
        json.dump(sample_mesh, f, indent=2)
    
    # Sample training data
    sample_training = [
        {"text": "User: What is quantum computing?\nAssistant: Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations."},
        {"text": "User: Explain neural networks\nAssistant: Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information."},
        {"text": "User: How does machine learning work?\nAssistant: Machine learning enables computers to learn from data without being explicitly programmed, improving performance through experience."}
    ]
    
    training_file = Path("data/training/user_demo_user_data.jsonl")
    with open(training_file, 'w') as f:
        for sample in sample_training:
            f.write(json.dumps(sample) + "\n")
    
    print(f"{GREEN}✓ Sample data created{RESET}")

def test_installation():
    """Test the installation."""
    print(f"\n{BLUE}Testing installation...{RESET}")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test imports
    try:
        from python.core import SaigonInference
        print(f"{GREEN}✓ Core modules importable{RESET}")
        tests_passed += 1
    except ImportError as e:
        print(f"{RED}✗ Failed to import core modules: {e}{RESET}")
        tests_failed += 1
    
    try:
        from python.training import AdapterTrainer
        print(f"{GREEN}✓ Training modules importable{RESET}")
        tests_passed += 1
    except ImportError as e:
        print(f"{RED}✗ Failed to import training modules: {e}{RESET}")
        tests_failed += 1
    
    # Test API
    try:
        from api.saigon_inference_api_v5 import app
        print(f"{GREEN}✓ API module importable{RESET}")
        tests_passed += 1
    except ImportError as e:
        print(f"{RED}✗ Failed to import API: {e}{RESET}")
        tests_failed += 1
    
    print(f"\n{CYAN}Test Results: {GREEN}{tests_passed} passed{RESET}, {RED}{tests_failed} failed{RESET}")
    
    return tests_failed == 0

def print_next_steps():
    """Print next steps for the user."""
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{GREEN}{BOLD}Setup Complete!{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    
    print(f"\n{YELLOW}Next Steps:{RESET}")
    print(f"1. Start the API server:")
    print(f"   {CYAN}python api/saigon_inference_api_v5.py{RESET}")
    print()
    print(f"2. Run the interactive demo:")
    print(f"   {CYAN}python scripts/demo_inference_v5.py --mode interactive{RESET}")
    print()
    print(f"3. Train a custom adapter:")
    print(f"   {CYAN}python python/training/train_lora_adapter_v5.py --user_id demo_user --dataset data/training/user_demo_user_data.jsonl{RESET}")
    print()
    print(f"4. Start the frontend (in another terminal):")
    print(f"   {CYAN}cd frontend/hybrid && npm run dev{RESET}")
    print()
    print(f"{GREEN}Documentation:{RESET} See README.md for detailed usage")
    print(f"{GREEN}API Docs:{RESET} http://localhost:8001/docs (after starting server)")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="TORI/Saigon Setup Script")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend setup")
    parser.add_argument("--skip-model", action="store_true", help="Skip model download")
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    
    args = parser.parse_args()
    
    if not args.test_only:
        print_banner()
        
        # Check Python version
        if not check_python_version():
            sys.exit(1)
        
        # Create directories
        create_directories()
        
        # Install dependencies
        if not args.skip_deps:
            if not install_dependencies():
                print(f"{YELLOW}⚠ Dependency installation failed. Continuing...{RESET}")
        
        # Check CUDA after installing PyTorch
        check_cuda()
        
        # Initialize metadata
        initialize_metadata()
        
        # Setup frontend
        if not args.skip_frontend:
            setup_frontend()
        
        # Create .env file
        create_env_file()
        
        # Download base model
        if not args.skip_model:
            download_base_model()
        
        # Create sample data
        create_sample_data()
    
    # Test installation
    if test_installation():
        print_next_steps()
    else:
        print(f"\n{RED}⚠ Some tests failed. Check the errors above.{RESET}")

if __name__ == "__main__":
    main()
