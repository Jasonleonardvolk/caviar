#!/usr/bin/env python3
"""
Setup code quality tools for the TORI/KHA project
Installs and configures linting, formatting, and pre-commit hooks
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd: list, check=True):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def setup_quality_tools():
    """Install and configure code quality tools"""
    
    logger.info("🔧 Setting up code quality tools for TORI/KHA...")
    logger.info("="*70)
    
    # Step 1: Install development dependencies
    logger.info("\n📦 Installing development dependencies...")
    
    dev_deps = [
        "black",       # Code formatter
        "flake8",      # Linter
        "mypy",        # Type checker
        "isort",       # Import sorter
        "pre-commit",  # Git hooks
        "pylint",      # Additional linting
        "autopep8",    # Auto-fix PEP8 issues
    ]
    
    for dep in dev_deps:
        logger.info(f"  Installing {dep}...")
        success, _ = run_command(["poetry", "add", "--group", "dev", dep])
        if success:
            logger.info(f"  ✅ {dep} installed")
        else:
            logger.warning(f"  ⚠️ Failed to install {dep}")
    
    # Step 2: Create configuration files
    logger.info("\n📝 Creating configuration files...")
    
    # flake8 configuration
    flake8_config = """[flake8]
max-line-length = 120
extend-ignore = E203, E266, E501, W503
max-complexity = 15
exclude = .git,__pycache__,build,dist,.venv,venv
"""
    
    with open(".flake8", "w") as f:
        f.write(flake8_config)
    logger.info("  ✅ Created .flake8")
    
    # mypy configuration
    mypy_config = """[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
ignore_missing_imports = True
"""
    
    with open("mypy.ini", "w") as f:
        f.write(mypy_config)
    logger.info("  ✅ Created mypy.ini")
    
    # pyproject.toml additions for tools
    pyproject_additions = """
[tool.black]
line-length = 120
target-version = ['py311']
include = '\\.pyi?$'

[tool.isort]
profile = "black"
line_length = 120
known_first_party = ["tori", "kha", "ingest_pdf", "python", "tools"]

[tool.pylint.messages_control]
disable = "C0330, C0326"

[tool.pylint.format]
max-line-length = "120"
"""
    
    # Step 3: Install pre-commit hooks
    logger.info("\n🪝 Setting up pre-commit hooks...")
    success, _ = run_command(["pre-commit", "install"])
    if success:
        logger.info("  ✅ Pre-commit hooks installed")
    else:
        logger.warning("  ⚠️ Failed to install pre-commit hooks")
    
    # Step 4: Create VS Code settings
    logger.info("\n💻 Creating VS Code settings...")
    
    vscode_settings = {
        "python.linting.enabled": True,
        "python.linting.flake8Enabled": True,
        "python.linting.pylintEnabled": False,
        "python.formatting.provider": "black",
        "python.formatting.blackArgs": ["--line-length=120"],
        "python.sortImports.args": ["--profile", "black"],
        "editor.formatOnSave": True,
        "editor.codeActionsOnSave": {
            "source.organizeImports": True
        },
        "[python]": {
            "editor.rulers": [120]
        }
    }
    
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    import json
    with open(vscode_dir / "settings.json", "w") as f:
        json.dump(vscode_settings, f, indent=2)
    logger.info("  ✅ Created .vscode/settings.json")
    
    # Step 5: Run initial checks
    logger.info("\n🔍 Running initial quality checks...")
    
    # Run black in check mode
    logger.info("  Running black...")
    success, output = run_command(["black", "--check", "--diff", "."], check=False)
    if not success:
        logger.info("  ⚠️ Some files need formatting. Run: black .")
    else:
        logger.info("  ✅ All files properly formatted")
    
    # Run flake8
    logger.info("  Running flake8...")
    success, output = run_command(["flake8", "."], check=False)
    if not success:
        logger.info("  ⚠️ Some linting issues found. Run: flake8")
    else:
        logger.info("  ✅ No linting issues")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ CODE QUALITY SETUP COMPLETE")
    logger.info("="*70)
    logger.info("\n📋 Next steps:")
    logger.info("1. Format all files: black .")
    logger.info("2. Sort imports: isort .")
    logger.info("3. Check types: mypy .")
    logger.info("4. Run linter: flake8")
    logger.info("5. Run pre-commit on all files: pre-commit run --all-files")
    logger.info("\n💡 Git hooks will now run automatically on commit!")


if __name__ == "__main__":
    setup_quality_tools()
