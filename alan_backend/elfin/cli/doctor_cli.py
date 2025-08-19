"""
Elf Doctor CLI Command.

Checks the environment for required dependencies and configurations for ELFIN.
"""

import click
import os
import sys
import subprocess
import importlib
import platform
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)


class DependencyCheck:
    """Base class for dependency checks."""
    
    def __init__(self, name: str, required: bool = True):
        self.name = name
        self.required = required
        self.status = "UNCHECKED"
        self.version = None
        self.details = None
    
    def check(self) -> bool:
        """
        Check if dependency is satisfied.
        
        Returns:
            True if dependency is satisfied, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_status_dict(self) -> Dict:
        """
        Get status dictionary.
        
        Returns:
            Dictionary containing check status information
        """
        return {
            "name": self.name,
            "required": self.required,
            "status": self.status,
            "version": self.version,
            "details": self.details
        }


class PythonModuleCheck(DependencyCheck):
    """Check for Python module dependency."""
    
    def __init__(self, name: str, module_name: str, required: bool = True, 
                min_version: Optional[str] = None):
        super().__init__(name, required)
        self.module_name = module_name
        self.min_version = min_version
    
    def check(self) -> bool:
        try:
            module = importlib.import_module(self.module_name)
            
            # Try to get version
            try:
                self.version = getattr(module, "__version__", None)
                if self.version is None and hasattr(module, "version"):
                    self.version = module.version
                if self.version is None and hasattr(module, "VERSION"):
                    self.version = module.VERSION
            except Exception:
                self.version = "Unknown"
            
            # Check minimum version if specified
            if self.min_version is not None and self.version is not None:
                if self._compare_versions(self.version, self.min_version) < 0:
                    self.status = "WARNING"
                    self.details = f"Version {self.version} < {self.min_version}"
                    return False
            
            self.status = "OK"
            return True
        except ImportError as e:
            self.status = "MISSING" if self.required else "OPTIONAL"
            self.details = str(e)
            return False
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """
        Compare version strings.
        
        Args:
            v1: First version string
            v2: Second version string
            
        Returns:
            -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
        """
        from packaging import version
        return (version.parse(v1) > version.parse(v2)) - (version.parse(v1) < version.parse(v2))


class CommandCheck(DependencyCheck):
    """Check for command-line tool dependency."""
    
    def __init__(self, name: str, command: str, version_flag: str = "--version", 
                required: bool = True, min_version: Optional[str] = None,
                version_extract: Optional[str] = None):
        super().__init__(name, required)
        self.command = command
        self.version_flag = version_flag
        self.min_version = min_version
        self.version_extract = version_extract
    
    def check(self) -> bool:
        # Check if command exists
        if not shutil.which(self.command):
            self.status = "MISSING" if self.required else "OPTIONAL"
            self.details = f"Command '{self.command}' not found in PATH"
            return False
        
        # Try to get version
        try:
            result = subprocess.run(
                [self.command, self.version_flag], 
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                self.status = "WARNING"
                self.details = f"Command exists but returned error code {result.returncode}"
                return False
            
            # Extract version from output
            output = result.stdout.strip() or result.stderr.strip()
            
            if self.version_extract:
                import re
                match = re.search(self.version_extract, output)
                if match:
                    self.version = match.group(1)
                else:
                    self.version = "Unknown"
            else:
                self.version = output.split("\n")[0]
            
            # Check minimum version if specified
            if self.min_version is not None and self.version is not None:
                # This is a simplistic version check that may not work for all formats
                # You might want to use the packaging module as in PythonModuleCheck
                if self.version < self.min_version:
                    self.status = "WARNING"
                    self.details = f"Version {self.version} < {self.min_version}"
                    return False
            
            self.status = "OK"
            return True
        except (subprocess.SubprocessError, OSError) as e:
            self.status = "ERROR"
            self.details = str(e)
            return False


class FileCheck(DependencyCheck):
    """Check for file existence."""
    
    def __init__(self, name: str, file_path: Union[str, Path], required: bool = True):
        super().__init__(name, required)
        self.file_path = Path(file_path)
    
    def check(self) -> bool:
        if self.file_path.exists():
            self.status = "OK"
            self.details = str(self.file_path.absolute())
            return True
        else:
            self.status = "MISSING" if self.required else "OPTIONAL"
            self.details = f"File not found: {self.file_path}"
            return False


class EnvironmentVariableCheck(DependencyCheck):
    """Check for environment variable existence."""
    
    def __init__(self, name: str, env_var: str, required: bool = True, 
                expected_value: Optional[str] = None):
        super().__init__(name, required)
        self.env_var = env_var
        self.expected_value = expected_value
    
    def check(self) -> bool:
        if self.env_var in os.environ:
            value = os.environ[self.env_var]
            
            if self.expected_value is not None and value != self.expected_value:
                self.status = "WARNING"
                self.details = f"Value is '{value}', expected '{self.expected_value}'"
                return False
            
            self.status = "OK"
            self.details = value if len(value) < 50 else value[:47] + "..."
            return True
        else:
            self.status = "MISSING" if self.required else "OPTIONAL"
            self.details = f"Environment variable {self.env_var} not set"
            return False


class GPUCheck(DependencyCheck):
    """Check for GPU availability."""
    
    def __init__(self, name: str = "GPU Support", required: bool = False):
        super().__init__(name, required)
    
    def check(self) -> bool:
        # Try different GPU detection methods
        
        # 1. Try CUDA through cupy
        try:
            import cupy as cp
            count = cp.cuda.runtime.getDeviceCount()
            if count > 0:
                # Get details of first GPU
                device = cp.cuda.runtime.getDeviceProperties(0)
                self.status = "OK"
                self.version = f"CUDA {cp.cuda.runtime.runtimeGetVersion() // 1000}.{(cp.cuda.runtime.runtimeGetVersion() % 1000) // 10}"
                self.details = f"Found {count} GPU(s). First GPU: {device['name'].decode()}"
                return True
        except ImportError:
            pass
        except Exception as e:
            self.details = f"cupy check failed: {str(e)}"
        
        # 2. Try CUDA through torch
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                self.status = "OK"
                self.version = f"CUDA {torch.version.cuda}"
                self.details = f"Found {count} GPU(s). First GPU: {torch.cuda.get_device_name(0)}"
                return True
        except ImportError:
            pass
        except Exception as e:
            if self.details:
                self.details += f"; torch check failed: {str(e)}"
            else:
                self.details = f"torch check failed: {str(e)}"
        
        # 3. Try TensorFlow
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.status = "OK"
                self.version = f"TensorFlow {tf.__version__}"
                self.details = f"Found {len(gpus)} GPU(s) through TensorFlow"
                return True
        except ImportError:
            pass
        except Exception as e:
            if self.details:
                self.details += f"; tensorflow check failed: {str(e)}"
            else:
                self.details = f"tensorflow check failed: {str(e)}"
        
        # No GPU found
        self.status = "MISSING" if self.required else "OPTIONAL"
        if not self.details:
            self.details = "No GPU detected"
        return False


class MOSEKCheck(DependencyCheck):
    """Check for MOSEK license and installation."""
    
    def __init__(self, name: str = "MOSEK", required: bool = False):
        super().__init__(name, required)
    
    def check(self) -> bool:
        # Check for mosek module
        try:
            import mosek
            self.version = mosek.Env().getversion()
        except ImportError:
            self.status = "MISSING" if self.required else "OPTIONAL"
            self.details = "MOSEK Python module not installed"
            return False
        except Exception as e:
            self.status = "ERROR"
            self.details = f"MOSEK installation error: {str(e)}"
            return False
        
        # Check for license file
        license_paths = [
            Path.home() / "mosek" / "mosek.lic",
            Path("/opt/mosek/mosek.lic"),
            Path(os.environ.get("MOSEKLM_LICENSE_FILE", "")),
        ]
        
        for path in license_paths:
            if path.exists():
                self.status = "OK"
                self.details = f"License found at {path}"
                return True
        
        self.status = "WARNING"
        self.details = "MOSEK installed but no license file found"
        return False


def check_system_info():
    """Get system information."""
    system_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_path": sys.executable,
        "cwd": os.getcwd(),
    }
    
    # Try to get physical memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        system_info["total_memory_gb"] = round(memory.total / (1024**3), 1)
        system_info["available_memory_gb"] = round(memory.available / (1024**3), 1)
    except ImportError:
        pass
    
    return system_info


def print_results(checks, system_info):
    """Print check results in a formatted way."""
    # Print system info
    click.secho("=== System Information ===", bold=True)
    click.echo(f"Python: {system_info['python_version'].splitlines()[0]}")
    click.echo(f"Platform: {system_info['platform']}")
    if "total_memory_gb" in system_info:
        click.echo(f"Memory: {system_info['available_memory_gb']} GB free / {system_info['total_memory_gb']} GB total")
    click.echo(f"Current Directory: {system_info['cwd']}")
    click.echo("")
    
    # Print dependency check results
    click.secho("=== Dependency Check Results ===", bold=True)
    
    failures = []
    warnings = []
    
    for check in checks:
        status = check.status
        name = check.name
        version = check.version or ""
        details = check.details or ""
        
        if status == "OK":
            status_str = click.style("✓", fg="green")
        elif status == "WARNING":
            status_str = click.style("⚠", fg="yellow")
            warnings.append(check)
        elif status == "MISSING" and check.required:
            status_str = click.style("✗", fg="red")
            failures.append(check)
        elif status == "ERROR":
            status_str = click.style("✗", fg="red")
            failures.append(check)
        else:
            status_str = click.style("○", fg="blue")
        
        click.echo(f"{status_str} {name:25} {version:15} {details}")
    
    # Summary
    click.echo("")
    
    if failures:
        click.secho("Required dependencies missing:", fg="red")
        for check in failures:
            click.echo(f"  - {check.name}: {check.details}")
    
    if warnings:
        click.secho("Warnings:", fg="yellow")
        for check in warnings:
            click.echo(f"  - {check.name}: {check.details}")
    
    if not failures:
        if warnings:
            click.secho("System check passed with warnings.", fg="yellow")
        else:
            click.secho("All dependency checks passed!", fg="green")
    else:
        click.secho(f"System check failed with {len(failures)} missing dependencies.", fg="red")
    
    return len(failures)


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
@click.option("--json", "json_output", is_flag=True, help="Output results as JSON")
def doctor(verbose, json_output):
    """
    Check environment for ELFIN system requirements.
    
    This command verifies that all required dependencies for running ELFIN
    are installed and configured correctly.
    """
    checks = [
        # Core Python dependencies
        PythonModuleCheck("NumPy", "numpy", min_version="1.20.0"),
        PythonModuleCheck("SciPy", "scipy", min_version="1.7.0"),
        PythonModuleCheck("Matplotlib", "matplotlib", min_version="3.4.0"),
        PythonModuleCheck("NetworkX", "networkx", min_version="2.6.0"),
        PythonModuleCheck("Click", "click", min_version="8.0.0"),
        
        # ELFIN-specific dependencies
        PythonModuleCheck("TOML", "toml"),
        PythonModuleCheck("PyYAML", "yaml", required=False),
        PythonModuleCheck("Jinja2", "jinja2", required=False),
        
        # ELFIN DSL parser dependencies
        PythonModuleCheck("Lark", "lark", min_version="1.0.0"),
        
        # Barrier/Lyapunov function dependencies
        PythonModuleCheck("SymPy", "sympy", required=False),
        MOSEKCheck(required=False),
        
        # Dashboard/visualization dependencies
        PythonModuleCheck("Plotly", "plotly", required=False),
        PythonModuleCheck("Dash", "dash", required=False),
        PythonModuleCheck("Flask", "flask", required=False),
        
        # Optional GPU support
        GPUCheck(required=False),
        
        # Development tools
        CommandCheck("Git", "git", version_extract=r"git version ([\d\.]+)"),
        CommandCheck("Poetry", "poetry", required=False),
        PythonModuleCheck("Pytest", "pytest", required=False),
        
        # Environment variables
        EnvironmentVariableCheck("Python Path", "PYTHONPATH", required=False),
    ]
    
    # Run all checks
    for check in checks:
        check.check()
    
    # Get system info
    system_info = check_system_info()
    
    # Output results
    if json_output:
        import json
        result = {
            "system_info": system_info,
            "checks": [check.get_status_dict() for check in checks]
        }
        click.echo(json.dumps(result, indent=2))
    else:
        failure_count = print_results(checks, system_info)
        
        if failure_count > 0:
            sys.exit(1)


if __name__ == "__main__":
    doctor()
