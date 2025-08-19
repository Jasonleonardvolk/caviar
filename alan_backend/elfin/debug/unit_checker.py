"""
Unit Checker for ELFIN

This module provides dimensional analysis and unit checking for ELFIN code,
detecting unit inconsistencies and suggesting fixes.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union

from alan_backend.elfin.units.units import UnitRegistry, UnitError, DimensionalityError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("elfin.debug.unit_checker")

class UnitDiagnostic:
    """Represents a unit consistency issue with suggested fixes."""
    
    def __init__(
        self, 
        message: str, 
        line: int, 
        column: int, 
        end_line: int = None, 
        end_column: int = None, 
        severity: str = "warning"
    ):
        """
        Initialize a unit diagnostic.
        
        Args:
            message: Description of the issue
            line: Line number (1-based)
            column: Column number (1-based)
            end_line: End line number (optional)
            end_column: End column number (optional)
            severity: Severity level ("error", "warning", "info")
        """
        self.message = message
        self.line = line
        self.column = column
        self.end_line = end_line or line
        self.end_column = end_column
        self.severity = severity
        self.fixes: List[UnitFix] = []
    
    def add_fix(self, fix: 'UnitFix'):
        """Add a suggested fix for this diagnostic."""
        self.fixes.append(fix)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic to a dictionary for LSP."""
        result = {
            "message": self.message,
            "range": {
                "start": {"line": self.line - 1, "character": self.column - 1},
                "end": {"line": self.end_line - 1, "character": self.end_column or 80}
            },
            "severity": self._get_severity_number(),
            "source": "elfin-unit-checker"
        }
        
        if self.fixes:
            result["fixes"] = [fix.to_dict() for fix in self.fixes]
        
        return result
    
    def _get_severity_number(self) -> int:
        """Convert severity string to LSP severity number."""
        severities = {
            "error": 1,
            "warning": 2,
            "info": 3,
            "hint": 4
        }
        return severities.get(self.severity.lower(), 2)


class UnitFix:
    """Represents a fix for a unit consistency issue."""
    
    def __init__(
        self, 
        title: str, 
        line: int, 
        column: int, 
        end_line: int, 
        end_column: int, 
        new_text: str
    ):
        """
        Initialize a unit fix.
        
        Args:
            title: Title of the fix
            line: Line number for the fix
            column: Start column for the fix
            end_line: End line for the fix
            end_column: End column for the fix
            new_text: New text to replace the range
        """
        self.title = title
        self.line = line
        self.column = column
        self.end_line = end_line
        self.end_column = end_column
        self.new_text = new_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fix to a dictionary for LSP."""
        return {
            "title": self.title,
            "edit": {
                "range": {
                    "start": {"line": self.line - 1, "character": self.column - 1},
                    "end": {"line": self.end_line - 1, "character": self.end_column - 1}
                },
                "newText": self.new_text
            }
        }


class UnitChecker:
    """
    Checks ELFIN code for unit consistency.
    
    Analyzes the dimensional consistency of ELFIN code, detects unit mismatches,
    and suggests fixes.
    """
    
    def __init__(self):
        """Initialize the unit checker."""
        self.ureg = UnitRegistry()
        self.variable_units: Dict[str, Any] = {}
        self.diagnostics: List[UnitDiagnostic] = []
        
        # Regular expressions for extracting information
        self.assignment_regex = re.compile(r'(\w+)\s*=\s*(.*)')
        self.function_regex = re.compile(r'(\w+)\s*\((.*)\)')
        self.property_regex = re.compile(r'property\s+(\w+)\s*=\s*(.*)')
        self.unit_annotation_regex = re.compile(r'(\w+)\s*:\s*(\w+)')
    
    def clear(self):
        """Clear the checker state."""
        self.variable_units.clear()
        self.diagnostics.clear()
    
    def register_unit(self, name: str, unit_str: str):
        """
        Register a unit for a variable.
        
        Args:
            name: Variable name
            unit_str: Unit string (e.g., 'm', 's', 'kg')
        """
        try:
            unit = self.ureg.parse_units(unit_str)
            self.variable_units[name] = unit
            logger.info(f"Registered unit for {name}: {unit_str}")
        except Exception as e:
            logger.error(f"Error registering unit for {name}: {e}")
    
    def register_standard_units(self):
        """Register standard units for common variable names."""
        standard_units = {
            "position": "m",
            "velocity": "m/s",
            "acceleration": "m/s^2",
            "force": "N",
            "torque": "N*m",
            "angle": "rad",
            "angular_velocity": "rad/s",
            "angular_acceleration": "rad/s^2",
            "mass": "kg",
            "time": "s",
            "frequency": "Hz",
            "energy": "J",
            "power": "W",
            "current": "A",
            "voltage": "V",
            "resistance": "ohm",
            "temperature": "K",
            "pressure": "Pa"
        }
        
        for name, unit in standard_units.items():
            self.register_unit(name, unit)
    
    def analyze_file(self, file_path: str) -> List[UnitDiagnostic]:
        """
        Analyze an ELFIN file for unit consistency.
        
        Args:
            file_path: Path to the ELFIN file
            
        Returns:
            List of unit diagnostics
        """
        self.clear()
        
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            return self.analyze_code(code)
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return []
    
    def analyze_code(self, code: str) -> List[UnitDiagnostic]:
        """
        Analyze ELFIN code for unit consistency.
        
        Args:
            code: ELFIN code to analyze
            
        Returns:
            List of unit diagnostics
        """
        self.clear()
        
        # Register standard units
        self.register_standard_units()
        
        # Extract variable annotations
        self._extract_unit_annotations(code)
        
        # Check assignments
        self._check_assignments(code)
        
        # Check function calls
        self._check_function_calls(code)
        
        return self.diagnostics
    
    def _extract_unit_annotations(self, code: str):
        """
        Extract unit annotations from the code.
        
        Args:
            code: ELFIN code to analyze
        """
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            matches = self.unit_annotation_regex.finditer(line)
            
            for match in matches:
                var_name = match.group(1)
                unit_name = match.group(2)
                self.register_unit(var_name, unit_name)
    
    def _check_assignments(self, code: str):
        """
        Check assignments for unit consistency.
        
        Args:
            code: ELFIN code to analyze
        """
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Skip comments and empty lines
            if line.strip().startswith('//') or not line.strip():
                continue
            
            # Check for assignments
            match = self.assignment_regex.search(line)
            if match:
                lhs = match.group(1).strip()
                rhs = match.group(2).strip()
                
                # Skip if it's a property definition
                if line.strip().startswith('property'):
                    continue
                
                self._check_assignment_units(lhs, rhs, line_num, match.start(1), match.end(2))
    
    def _check_assignment_units(self, lhs: str, rhs: str, line: int, start_col: int, end_col: int):
        """
        Check units in an assignment.
        
        Args:
            lhs: Left-hand side of the assignment
            rhs: Right-hand side of the assignment
            line: Line number
            start_col: Start column
            end_col: End column
        """
        # Skip if we don't know the units of the LHS
        if lhs not in self.variable_units:
            return
        
        lhs_unit = self.variable_units[lhs]
        
        # Simple expression handling (not a full parser)
        # In a real implementation, we'd use a proper parser
        try:
            # Check for direct variable assignments
            if rhs in self.variable_units:
                rhs_unit = self.variable_units[rhs]
                
                # Check for unit mismatch
                try:
                    # Try to convert to the target unit
                    lhs_unit.to(rhs_unit)
                except DimensionalityError:
                    # Units are incompatible
                    diagnostic = UnitDiagnostic(
                        f"Unit mismatch: {lhs} ({lhs_unit}) = {rhs} ({rhs_unit})",
                        line,
                        start_col + 1,
                        line,
                        end_col + 1,
                        "warning"
                    )
                    
                    # Add a fix: add conversion factor
                    conversion_text = self._generate_conversion_fix(rhs, rhs_unit, lhs_unit)
                    if conversion_text:
                        fix = UnitFix(
                            f"Convert {rhs} to {lhs_unit}",
                            line,
                            start_col + 1 + len(lhs) + 1,  # Position after '='
                            line,
                            end_col + 1,
                            conversion_text
                        )
                        diagnostic.add_fix(fix)
                    
                    self.diagnostics.append(diagnostic)
        except Exception as e:
            logger.debug(f"Error checking assignment units: {e}")
    
    def _check_function_calls(self, code: str):
        """
        Check function calls for unit consistency.
        
        Args:
            code: ELFIN code to analyze
        """
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Skip comments and empty lines
            if line.strip().startswith('//') or not line.strip():
                continue
            
            # Check for function calls
            match = self.function_regex.search(line)
            if match:
                func_name = match.group(1).strip()
                args = match.group(2).strip()
                
                # For now, just log it
                logger.debug(f"Found function call: {func_name}({args})")
                
                # In a real implementation, we'd check argument units
                # based on function signatures
    
    def _generate_conversion_fix(self, var_name: str, from_unit: Any, to_unit: Any) -> Optional[str]:
        """
        Generate code to convert between units.
        
        Args:
            var_name: Variable name
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Conversion code or None if not possible
        """
        try:
            # Try to find a conversion factor
            if from_unit.dimensionality == to_unit.dimensionality:
                # Units are compatible, just need conversion factor
                # Calculate the conversion factor
                try:
                    factor = from_unit.to(to_unit).magnitude
                    if factor == 1.0:
                        return f"{var_name}"  # No conversion needed
                    else:
                        return f"{var_name} * {factor}"
                except Exception:
                    # Can't determine exact factor, use unit conversion function
                    return f"convert({var_name}, '{from_unit}', '{to_unit}')"
            else:
                # Units have different dimensions
                # In a real implementation, we might suggest more sophisticated fixes
                return None
        except Exception as e:
            logger.debug(f"Error generating conversion fix: {e}")
            return None


# Create a singleton instance
unit_checker = UnitChecker()
