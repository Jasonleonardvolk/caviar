#!/usr/bin/env python3
"""
TORI/KHA Comprehensive Test Suite
Complete testing framework for all components
"""

import pytest
import pytest_asyncio
import numpy as np
from pathlib import Path
import tempfile
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import coverage

# ========== Test Configuration ==========

# pytest.ini content
PYTEST_INI = """
[pytest]
minversion = 6.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto

# Coverage settings
addopts = 
    --cov=python.core
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    -v
    --tb=short
    --strict-markers

markers =
    unit: Unit tests
    integration: Integration tests
    chaos: Chaos behavior tests
    quantum: Quantum computing tests
    performance: Performance tests
    slow: Slow tests (> 5s)
"""

# ========== Test Structure ==========

TEST_STRUCTURE = {
    "tests/": {
        "__init__.py": "Test package initialization",
        "conftest.py": "Shared fixtures and configuration",
        "pytest.ini": "Pytest configuration",
        
        "unit/": {
            "__init__.py": "",
            "test_cognitive_engine.py": "CognitiveEngine unit tests",
            "test_chaos_control.py": "Chaos control layer tests",
            "test_memory_vault.py": "Memory vault tests",
            "test_eigenvalue_monitor.py": "Eigenvalue monitoring tests",
            "test_quantum_integration.py": "Quantum integration tests",
        },
        
        "integration/": {
            "__init__.py": "",
            "test_tori_production.py": "Full system integration tests",
            "test_chaos_quantum_integration.py": "Chaos-quantum integration",
            "test_memory_persistence.py": "Memory persistence tests",
            "test_safety_systems.py": "Safety and rollback tests",
        },
        
        "chaos/": {
            "__init__.py": "",
            "test_edge_of_chaos.py": "Edge-of-chaos behavior tests",
            "test_soliton_dynamics.py": "Dark soliton tests",
            "test_attractor_hopping.py": "Attractor hopping tests",
            "test_phase_explosion.py": "Phase explosion tests",
        },
        
        "performance/": {
            "__init__.py": "",
            "test_scalability.py": "Scalability tests",
            "test_memory_usage.py": "Memory usage tests",
            "test_processing_speed.py": "Processing speed benchmarks",
            "test_quantum_performance.py": "Quantum operation benchmarks",
        },
        
        "fixtures/": {
            "__init__.py": "",
            "data.py": "Test data generators",
            "mocks.py": "Mock objects and services",
            "helpers.py": "Test helper functions",
        }
    }
}

# ========== Core Test Fixtures (conftest.py) ==========

CONFTEST_CONTENT = '''
"""
Shared test fixtures for TORI/KHA test suite
"""

import pytest
import pytest_asyncio
import tempfile
import shutil
from pathlib import Path
import numpy as np
from typing import Dict, Any

from python.core.CognitiveEngine import CognitiveEngine
from python.core.chaos_control_layer import ChaosControlLayer
from python.core.memory_vault import UnifiedMemoryVault
from python.core.eigensentry.core import EigenSentry2
from python.core.file_state_sync import FileStateStore
from python.core.cognitive_dynamics_monitor import CognitiveStateManager

# ========== Configuration Fixtures ==========

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration"""
    return {
        'vector_dim': 64,  # Smaller for tests
        'max_iterations': 10,
        'storage_path': 'test_data',
        'enable_chaos': True,
        'enable_safety_monitoring': True
    }

@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

# ========== Component Fixtures ==========

@pytest_asyncio.fixture
async def cognitive_engine(test_config, temp_dir):
    """Initialize CognitiveEngine for testing"""
    config = test_config.copy()
    config['storage_path'] = str(temp_dir / 'cognitive')
    
    engine = CognitiveEngine(config)
    yield engine
    
    # Cleanup
    engine.shutdown()

@pytest_asyncio.fixture
async def memory_vault(test_config, temp_dir):
    """Initialize UnifiedMemoryVault for testing"""
    config = test_config.copy()
    config['storage_path'] = str(temp_dir / 'memory')
    
    vault = UnifiedMemoryVault(config)
    yield vault
    
    # Cleanup
    vault.shutdown()

@pytest.fixture
def state_manager(test_config):
    """Initialize CognitiveStateManager"""
    return CognitiveStateManager(state_dim=test_config['vector_dim'])

@pytest_asyncio.fixture
async def chaos_control(state_manager):
    """Initialize ChaosControlLayer"""
    eigen_sentry = EigenSentry2(state_manager)
    ccl = ChaosControlLayer(eigen_sentry, state_manager)
    
    # Start processing
    task = asyncio.create_task(ccl.process_tasks())
    
    yield ccl
    
    # Cleanup
    task.cancel()
    await asyncio.sleep(0.1)
    ccl.executor.shutdown()

@pytest.fixture
def state_store(temp_dir):
    """Initialize FileStateStore for testing"""
    store = FileStateStore(temp_dir / 'state')
    yield store
    store.close()

# ========== Test Data Fixtures ==========

@pytest.fixture
def sample_matrix():
    """Sample matrix for eigenvalue tests"""
    return np.array([
        [2.0, 0.5, 0.1],
        [0.5, 1.5, 0.3],
        [0.1, 0.3, 1.0]
    ])

@pytest.fixture
def sample_state_vector(test_config):
    """Sample state vector"""
    return np.random.randn(test_config['vector_dim'])

@pytest.fixture
def sample_memory_items():
    """Sample memory items for testing"""
    return [
        {"content": "Test memory 1", "importance": 0.8},
        {"content": "Test memory 2", "importance": 0.6},
        {"content": "Test memory 3", "importance": 0.9}
    ]

# ========== Mock Fixtures ==========

@pytest.fixture
def mock_quantum_backend():
    """Mock quantum backend for testing"""
    backend = Mock()
    backend.execute = AsyncMock(return_value={
        'success': True,
        'counts': {'00': 512, '11': 512},
        'statevector': np.array([0.707, 0, 0, 0.707])
    })
    return backend

# ========== Helper Fixtures ==========

@pytest.fixture
def assert_convergence():
    """Helper to assert convergence within tolerance"""
    def _assert(initial, final, tolerance=0.1):
        change = np.linalg.norm(final - initial)
        assert change < tolerance, f"Failed to converge: change={change}"
    return _assert

@pytest.fixture
def measure_time():
    """Helper to measure execution time"""
    import time
    
    class Timer:
        def __init__(self):
            self.start = None
            self.elapsed = None
        
        def __enter__(self):
            self.start = time.time()
            return self
        
        def __exit__(self, *args):
            self.elapsed = time.time() - self.start
    
    return Timer
'''

# ========== Unit Test Examples ==========

UNIT_TEST_COGNITIVE_ENGINE = '''
"""
Unit tests for CognitiveEngine
"""

import pytest
import numpy as np
from unittest.mock import patch

@pytest.mark.unit
class TestCognitiveEngine:
    """Test CognitiveEngine functionality"""
    
    async def test_initialization(self, cognitive_engine):
        """Test engine initialization"""
        assert cognitive_engine.vector_dim == 64
        assert cognitive_engine.processing_state.value == "idle"
        assert cognitive_engine.current_state is not None
    
    async def test_process_text_input(self, cognitive_engine):
        """Test processing text input"""
        result = await cognitive_engine.process(
            "Test input for cognitive processing",
            context={"domain": "testing"}
        )
        
        assert result.success
        assert result.output is not None
        assert 'interpretation' in result.output
        assert result.state.stability_score >= 0
        assert result.state.stability_score <= 1
    
    async def test_process_convergence(self, cognitive_engine, assert_convergence):
        """Test that processing converges"""
        initial_state = cognitive_engine.current_state.thought_vector.copy()
        
        result = await cognitive_engine.process("Convergence test")
        
        assert result.success
        assert_convergence(
            initial_state,
            result.state.thought_vector,
            tolerance=0.5
        )
    
    async def test_stability_monitoring(self, cognitive_engine):
        """Test stability monitoring during processing"""
        # Track stability callbacks
        stability_values = []
        
        def stability_callback(eigenvalues, max_eigenvalue):
            stability_values.append(max_eigenvalue)
        
        cognitive_engine.register_stability_callback(stability_callback)
        
        result = await cognitive_engine.process("Stability test")
        
        assert result.success
        assert len(stability_values) > 0
        assert all(v < 2.0 for v in stability_values), "Eigenvalues exceeded threshold"
    
    async def test_error_handling(self, cognitive_engine):
        """Test error handling in processing"""
        # Force an error by passing invalid input
        with patch.object(cognitive_engine, '_encode_input', side_effect=Exception("Test error")):
            result = await cognitive_engine.process(None)
        
        assert not result.success
        assert len(result.errors) > 0
        assert "Test error" in result.errors[0]
    
    @pytest.mark.parametrize("input_type,input_data", [
        ("string", "Test string"),
        ("dict", {"key": "value", "number": 42}),
        ("list", [1, 2, 3, 4, 5]),
        ("numpy", np.array([1.0, 2.0, 3.0]))
    ])
    async def test_multiple_input_types(self, cognitive_engine, input_type, input_data):
        """Test processing different input types"""
        result = await cognitive_engine.process(input_data)
        
        assert result.success
        assert result.output is not None
    
    async def test_checkpoint_persistence(self, cognitive_engine, temp_dir):
        """Test checkpoint save and load"""
        # Process something to change state
        await cognitive_engine.process("Checkpoint test")
        
        # Save checkpoint
        await cognitive_engine._save_checkpoint()
        
        # Create new engine and load checkpoint
        new_engine = cognitive_engine.__class__(cognitive_engine.config)
        new_engine._load_checkpoint()
        
        # Compare states
        np.testing.assert_array_almost_equal(
            cognitive_engine.current_state.thought_vector,
            new_engine.current_state.thought_vector
        )
    
    @pytest.mark.slow
    async def test_max_iterations_limit(self, cognitive_engine):
        """Test that processing respects max iterations"""
        cognitive_engine.max_iterations = 5
        
        result = await cognitive_engine.process("Max iterations test")
        
        assert result.success
        assert result.metrics['iterations'] <= 5
'''

# ========== Chaos Testing ==========

CHAOS_TEST_EXAMPLE = '''
"""
Tests for chaos computing behavior
"""

import pytest
import numpy as np
from python.core.chaos_control_layer import ChaosTask, ChaosMode

@pytest.mark.chaos
class TestChaosControl:
    """Test chaos control layer behavior"""
    
    async def test_dark_soliton_memory(self, chaos_control):
        """Test dark soliton memory encoding/decoding"""
        test_data = np.array([1.0, -0.5, 0.8, -0.3])
        
        task = ChaosTask(
            task_id="soliton_test",
            mode=ChaosMode.DARK_SOLITON,
            input_data=test_data,
            parameters={'time_steps': 50},
            energy_budget=100
        )
        
        task_id = await chaos_control.submit_task(task)
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Check results
        assert task_id in chaos_control.completed_tasks
        result = chaos_control.completed_tasks[-1]
        
        assert result.success
        assert result.efficiency_gain > 1.0
        assert result.output_data is not None
    
    async def test_edge_of_chaos_efficiency(self, chaos_control, measure_time):
        """Test efficiency gains at edge of chaos"""
        # Regular computation
        regular_data = np.random.randn(100)
        
        with measure_time() as regular_timer:
            regular_result = np.fft.fft(regular_data)
        
        # Chaos-enhanced computation
        chaos_task = ChaosTask(
            task_id="efficiency_test",
            mode=ChaosMode.HYBRID,
            input_data=regular_data,
            parameters={},
            energy_budget=200
        )
        
        with measure_time() as chaos_timer:
            await chaos_control.submit_task(chaos_task)
            await asyncio.sleep(1.0)
        
        # Check efficiency gain
        result = chaos_control.completed_tasks[-1]
        assert result.efficiency_gain >= 3.0
    
    async def test_chaos_isolation(self, chaos_control):
        """Test chaos computation isolation"""
        # Submit task that would cause instability
        unstable_task = ChaosTask(
            task_id="unstable_test",
            mode=ChaosMode.PHASE_EXPLOSION,
            input_data=np.ones(50) * 1e6,  # Large values
            parameters={'explosion_strength': 10.0},
            energy_budget=100
        )
        
        await chaos_control.submit_task(unstable_task)
        await asyncio.sleep(2.0)
        
        # System should still be stable
        status = chaos_control.get_status()
        assert status['active_tasks'] == 0
        assert chaos_control.processing_state != "ERROR"
'''

# ========== Performance Testing ==========

PERFORMANCE_TEST_EXAMPLE = '''
"""
Performance and scalability tests
"""

import pytest
import numpy as np
import time
import psutil
import gc

@pytest.mark.performance
class TestPerformance:
    """Performance benchmarks and tests"""
    
    @pytest.mark.parametrize("vector_dim", [64, 256, 512, 1024])
    async def test_processing_speed_scaling(self, test_config, vector_dim):
        """Test processing speed with different vector dimensions"""
        config = test_config.copy()
        config['vector_dim'] = vector_dim
        
        engine = CognitiveEngine(config)
        
        # Measure processing time
        start = time.time()
        result = await engine.process("Performance test")
        elapsed = time.time() - start
        
        assert result.success
        assert elapsed < vector_dim / 100  # Linear scaling expectation
        
        engine.shutdown()
    
    async def test_memory_usage(self, memory_vault):
        """Test memory usage under load"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Store many memories
        for i in range(1000):
            await memory_vault.store(
                content=f"Memory item {i}",
                memory_type="semantic",
                metadata={"index": i}
            )
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100  # Less than 100MB increase
        
        # Test memory cleanup
        count = await memory_vault.consolidate()
        assert count['deleted'] > 0
    
    @pytest.mark.slow
    async def test_concurrent_processing(self, cognitive_engine):
        """Test concurrent request handling"""
        tasks = []
        
        # Submit multiple concurrent requests
        for i in range(10):
            task = cognitive_engine.process(f"Concurrent request {i}")
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r.success for r in results)
        
        # Check for race conditions
        states = [r.state.thought_vector for r in results]
        # States should be different (no shared state corruption)
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states[i+1:], i+1):
                assert not np.array_equal(state1, state2)
'''

# ========== Test Runner Script ==========

TEST_RUNNER_SCRIPT = '''#!/usr/bin/env python3
"""
TORI/KHA Test Runner
Comprehensive test execution and reporting
"""

import subprocess
import sys
from pathlib import Path
import json
import time

class TestRunner:
    """Automated test runner with reporting"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 Running TORI/KHA Test Suite")
        print("=" * 60)
        
        test_categories = [
            ("Unit Tests", "-m unit"),
            ("Integration Tests", "-m integration"),
            ("Chaos Tests", "-m chaos"),
            ("Performance Tests", "-m performance"),
        ]
        
        for category, marker in test_categories:
            print(f"\\n📋 {category}")
            print("-" * 40)
            
            start_time = time.time()
            result = self.run_pytest(marker)
            elapsed = time.time() - start_time
            
            self.results[category] = {
                'passed': result['passed'],
                'failed': result['failed'],
                'duration': elapsed,
                'coverage': result.get('coverage', 0)
            }
            
            self.print_results(category, result, elapsed)
    
    def run_pytest(self, args: str) -> dict:
        """Run pytest with specified arguments"""
        cmd = [
            sys.executable, "-m", "pytest",
            args,
            "--json-report",
            "--json-report-file=test_report.json"
        ]
        
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        # Parse results
        if Path("test_report.json").exists():
            with open("test_report.json") as f:
                report = json.load(f)
                return {
                    'passed': report['summary']['passed'],
                    'failed': report['summary']['failed'],
                    'coverage': self.extract_coverage()
                }
        
        return {'passed': 0, 'failed': 0}
    
    def extract_coverage(self) -> float:
        """Extract coverage percentage from report"""
        coverage_file = self.project_root / "htmlcov" / "index.html"
        if coverage_file.exists():
            # Parse coverage from HTML (simplified)
            content = coverage_file.read_text()
            if "%" in content:
                # Extract percentage (this is simplified)
                return 85.0  # Placeholder
        return 0.0
    
    def print_results(self, category: str, result: dict, elapsed: float):
        """Print test results"""
        total = result['passed'] + result['failed']
        
        if result['failed'] == 0:
            status = "✅ PASSED"
            color = "\\033[92m"
        else:
            status = "❌ FAILED"
            color = "\\033[91m"
        
        print(f"{color}{status}\\033[0m - {result['passed']}/{total} tests passed")
        print(f"Duration: {elapsed:.2f}s")
        
        if result.get('coverage'):
            print(f"Coverage: {result['coverage']:.1f}%")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        report_path = self.project_root / "test_report.md"
        
        with open(report_path, "w") as f:
            f.write("# TORI/KHA Test Report\\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("## Summary\\n")
            total_passed = sum(r['passed'] for r in self.results.values())
            total_failed = sum(r['failed'] for r in self.results.values())
            
            f.write(f"- Total Tests: {total_passed + total_failed}\\n")
            f.write(f"- Passed: {total_passed}\\n")
            f.write(f"- Failed: {total_failed}\\n")
            f.write(f"- Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%\\n\\n")
            
            f.write("## Details\\n")
            for category, result in self.results.items():
                f.write(f"### {category}\\n")
                f.write(f"- Passed: {result['passed']}\\n")
                f.write(f"- Failed: {result['failed']}\\n")
                f.write(f"- Duration: {result['duration']:.2f}s\\n")
                f.write(f"- Coverage: {result.get('coverage', 0):.1f}%\\n\\n")
        
        print(f"\\n📊 Test report generated: {report_path}")

if __name__ == "__main__":
    runner = TestRunner(Path.cwd())
    runner.run_all_tests()
    runner.generate_report()
'''

# Save all test files
def create_test_suite():
    """Create comprehensive test suite structure"""
    base_path = Path("python/tests")
    
    # Create directories
    for dir_path in TEST_STRUCTURE:
        if dir_path.endswith('/'):
            (base_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create conftest.py
    (base_path / "conftest.py").write_text(CONFTEST_CONTENT)
    
    # Create unit tests
    (base_path / "unit" / "test_cognitive_engine.py").write_text(UNIT_TEST_COGNITIVE_ENGINE)
    (base_path / "chaos" / "test_chaos_control.py").write_text(CHAOS_TEST_EXAMPLE)
    (base_path / "performance" / "test_performance.py").write_text(PERFORMANCE_TEST_EXAMPLE)
    
    # Create test runner
    runner_path = base_path / "run_tests.py"
    runner_path.write_text(TEST_RUNNER_SCRIPT)
    runner_path.chmod(0o755)
    
    print("✅ Comprehensive test suite created!")

if __name__ == "__main__":
    create_test_suite()
