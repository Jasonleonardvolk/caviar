"""
ELFIN Module System Errors

This module defines the exception classes used by the ELFIN module system.
"""

class ModuleError(Exception):
    """Base class for all module system errors."""
    pass


class ModuleNotFoundError(ModuleError):
    """Raised when a module cannot be found in any of the search paths."""
    
    def __init__(self, module_name, search_paths=None):
        self.module_name = module_name
        self.search_paths = search_paths or []
        search_paths_str = '\n  - '.join([''] + [str(p) for p in self.search_paths])
        message = f"Module '{module_name}' not found in any of the search paths:{search_paths_str}"
        super().__init__(message)


class CircularDependencyError(ModuleError):
    """Raised when a circular dependency is detected in module imports."""
    
    def __init__(self, import_chain):
        self.import_chain = import_chain
        chain_str = ' -> '.join(import_chain)
        message = f"Circular dependency detected: {chain_str}"
        super().__init__(message)


class ModuleParseError(ModuleError):
    """Raised when a module cannot be parsed."""
    
    def __init__(self, module_path, original_error=None):
        self.module_path = module_path
        self.original_error = original_error
        message = f"Error parsing module '{module_path}'"
        if original_error:
            message += f": {str(original_error)}"
        super().__init__(message)
