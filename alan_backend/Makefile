# ALAN Banksy Core Makefile
# Provides build, test, and simulation commands for the ALAN system

.PHONY: all sim test qa flash clean

# Default target
all: sim

# Python environment setup
.python_env:
	@echo "Setting up Python environment..."
	pip install -r requirements.txt
	touch .python_env

# Simulation targets
sim: .python_env
	@echo "Running ALAN simulation..."
	python core/demo_banksy_alan.py

# Just run the oscillator demo
sim_oscillator: .python_env
	@echo "Running Banksy oscillator simulation..."
	python -c "from core.demo_banksy_alan import run_oscillator_demo; run_oscillator_demo()"

# Just run the TRS controller demo
sim_trs: .python_env
	@echo "Running TRS controller simulation..."
	python -c "from core.demo_banksy_alan import run_trs_controller_demo; run_trs_controller_demo()"

# Just run the Hopfield memory demo
sim_memory: .python_env
	@echo "Running Hopfield memory simulation..."
	python -c "from core.demo_banksy_alan import run_hopfield_memory_demo; run_hopfield_memory_demo()"

# Just run the full system demo
sim_full: .python_env
	@echo "Running full ALAN system simulation..."
	python -c "from core.demo_banksy_alan import run_full_system_demo; run_full_system_demo()"

# Testing targets
test: .python_env
	@echo "Running ALAN tests..."
	pytest core/tests -v

# Quality assurance
qa: .python_env
	@echo "Running code quality checks..."
	flake8 core
	mypy core
	black --check core

# Format code
format: .python_env
	@echo "Formatting code..."
	black core

# Hardware flash simulation (placeholder for future hardware support)
flash:
	@echo "Simulating hardware flash..."
	@echo "This is a placeholder for future ST-NO-8 board support."
	@echo "When hardware is available, this will flash the firmware."

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name ".mypy_cache" -type d -exec rm -rf {} +
	rm -f .python_env

# Show help
help:
	@echo "ALAN Banksy Core Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Default target, runs simulation"
	@echo "  sim        - Run full simulation demo"
	@echo "  sim_oscillator - Run oscillator component demo"
	@echo "  sim_trs    - Run TRS controller component demo"
	@echo "  sim_memory - Run Hopfield memory component demo"
	@echo "  sim_full   - Run full system demo"
	@echo "  test       - Run tests"
	@echo "  qa         - Run code quality checks"
	@echo "  format     - Format code with black"
	@echo "  flash      - Simulate hardware flash (placeholder)"
	@echo "  clean      - Clean build artifacts"
	@echo "  help       - Show this help message"
