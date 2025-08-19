"""
Unit validation and conversion for ELFIN bridge.

This module provides utilities for validating and converting units
at the boundary of the ELFIN language and the numeric ALAN core,
preventing subtle bugs from unit mismatches.
"""

import functools
import math
import re
import logging
from typing import Dict, Any, Callable, TypeVar, cast, Optional, Set, Tuple, List, Union

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar('T')


# Common unit conversion factors
# Expanded to include all frequency units with prefixes
UNIT_FACTORS = {
    # Angle conversions
    'rad': 1.0,
    'deg': math.pi / 180.0,
    'grad': math.pi / 200.0,
    'arcmin': math.pi / (180.0 * 60.0),
    'arcsec': math.pi / (180.0 * 3600.0),
    
    # Time conversions
    's': 1.0,
    'ms': 1.0e-3,
    'us': 1.0e-6,
    'ns': 1.0e-9,
    'min': 60.0,
    'hour': 3600.0,
    
    # Distance conversions
    'm': 1.0,
    'km': 1000.0,
    'cm': 0.01,
    'mm': 0.001,
    'um': 1.0e-6,
    'nm': 1.0e-9,
    
    # Temperature conversions (to Kelvin)
    'K': 1.0,
    'C': 1.0,  # Offset handled separately
    'F': 5.0/9.0,  # Offset handled separately
    
    # Energy conversions
    'J': 1.0,
    'kJ': 1000.0,
    'eV': 1.602176634e-19,
    'keV': 1.602176634e-16,
    'MeV': 1.602176634e-13,
    
    # Frequency conversions - complete set with all prefixes
    'Hz': 1.0,
    'kHz': 1.0e3,
    'MHz': 1.0e6,
    'GHz': 1.0e9,
    'THz': 1.0e12,
    'mHz': 1.0e-3,
    'uHz': 1.0e-6,
    
    # Angular frequency (rad/s) - complete set with all prefixes
    'rad/s': 1.0 / (2.0 * math.pi),  # Convert from rad/s to Hz (base unit)
    'rad/ms': 1.0e3 / (2.0 * math.pi),
    'rad/us': 1.0e6 / (2.0 * math.pi),
    'krad/s': 1.0e3 / (2.0 * math.pi),
    'Mrad/s': 1.0e6 / (2.0 * math.pi),
    'Grad/s': 1.0e9 / (2.0 * math.pi),
}

# Unit dimensions mapped to SI base units
# Each unit is represented as a vector of powers of base dimensions
# [angle, time, length, mass, temperature, current, luminous intensity]
UNIT_DIMENSIONS = {
    # Dimensionless
    'dimensionless': [0, 0, 0, 0, 0, 0, 0],
    '1': [0, 0, 0, 0, 0, 0, 0],
    
    # Angle units
    'rad': [1, 0, 0, 0, 0, 0, 0],
    'deg': [1, 0, 0, 0, 0, 0, 0],
    'grad': [1, 0, 0, 0, 0, 0, 0],
    'arcmin': [1, 0, 0, 0, 0, 0, 0],
    'arcsec': [1, 0, 0, 0, 0, 0, 0],
    
    # Time units
    's': [0, 1, 0, 0, 0, 0, 0],
    'ms': [0, 1, 0, 0, 0, 0, 0],
    'us': [0, 1, 0, 0, 0, 0, 0],
    'ns': [0, 1, 0, 0, 0, 0, 0],
    'min': [0, 1, 0, 0, 0, 0, 0],
    'hour': [0, 1, 0, 0, 0, 0, 0],
    
    # Length units
    'm': [0, 0, 1, 0, 0, 0, 0],
    'km': [0, 0, 1, 0, 0, 0, 0],
    'cm': [0, 0, 1, 0, 0, 0, 0],
    'mm': [0, 0, 1, 0, 0, 0, 0],
    'um': [0, 0, 1, 0, 0, 0, 0],
    'nm': [0, 0, 1, 0, 0, 0, 0],
    
    # Temperature units
    'K': [0, 0, 0, 0, 1, 0, 0],
    'C': [0, 0, 0, 0, 1, 0, 0],
    'F': [0, 0, 0, 0, 1, 0, 0],
    
    # Energy units
    'J': [0, -2, 2, 1, 0, 0, 0],  # kg⋅m²/s²
    'kJ': [0, -2, 2, 1, 0, 0, 0],
    'eV': [0, -2, 2, 1, 0, 0, 0],
    'keV': [0, -2, 2, 1, 0, 0, 0],
    'MeV': [0, -2, 2, 1, 0, 0, 0],
    
    # Frequency units - all map to s^-1
    'Hz': [0, -1, 0, 0, 0, 0, 0],
    'kHz': [0, -1, 0, 0, 0, 0, 0],
    'MHz': [0, -1, 0, 0, 0, 0, 0],
    'GHz': [0, -1, 0, 0, 0, 0, 0],
    'THz': [0, -1, 0, 0, 0, 0, 0],
    'mHz': [0, -1, 0, 0, 0, 0, 0],
    'uHz': [0, -1, 0, 0, 0, 0, 0],
    
    # Angular frequency - all include angle dimension
    'rad/s': [1, -1, 0, 0, 0, 0, 0],
    'rad/ms': [1, -1, 0, 0, 0, 0, 0],
    'rad/us': [1, -1, 0, 0, 0, 0, 0],
    'krad/s': [1, -1, 0, 0, 0, 0, 0],
    'Mrad/s': [1, -1, 0, 0, 0, 0, 0],
    'Grad/s': [1, -1, 0, 0, 0, 0, 0],
}

# Mapping of frequency units to their base unit types
FREQUENCY_UNITS = {
    'Hz': 'frequency',
    'kHz': 'frequency',
    'MHz': 'frequency',
    'GHz': 'frequency',
    'THz': 'frequency',
    'mHz': 'frequency',
    'uHz': 'frequency',
    'rad/s': 'angular_frequency',
    'rad/ms': 'angular_frequency',
    'rad/us': 'angular_frequency',
    'krad/s': 'angular_frequency',
    'Mrad/s': 'angular_frequency',
    'Grad/s': 'angular_frequency',
}


class UnitError(Exception):
    """Exception raised for unit validation errors."""
    pass


def parse_unit(unit_str: str) -> Tuple[float, str]:
    """Parse a unit string into factor and base unit.
    
    This function is exposed for CLI unit validation and feedback.
    
    Args:
        unit_str: The unit string (e.g., 'rad', 'deg', 'm/s')
        
    Returns:
        (factor, base_unit): The conversion factor and base unit
    
    Raises:
        UnitError: If the unit is not recognized
    """
    # Handle null unit
    if not unit_str or unit_str == 'dimensionless' or unit_str == '1':
        return 1.0, 'dimensionless'
    
    # Check if it's a basic unit
    if unit_str in UNIT_FACTORS:
        return UNIT_FACTORS[unit_str], unit_str.split('/')[-1]
    
    # Try to parse compound units (e.g., 'm/s')
    if '/' in unit_str:
        numerator, denominator = unit_str.split('/', 1)
        if numerator in UNIT_FACTORS and denominator in UNIT_FACTORS:
            return UNIT_FACTORS[numerator] / UNIT_FACTORS[denominator], f"{numerator}/{denominator}"
    
    # If we get here, we don't recognize the unit
    # For frequency units, try to suggest a close match
    if unit_str.endswith('Hz') or unit_str.endswith('rad/s'):
        suggestions = []
        if any(u.endswith('Hz') for u in UNIT_FACTORS.keys()):
            suggestions.append(f"try using one of: Hz, kHz, MHz, GHz, THz")
        if any(u.endswith('rad/s') for u in UNIT_FACTORS.keys()):
            suggestions.append(f"for angular frequency use: rad/s, krad/s, Mrad/s")
        
        suggest_str = "; ".join(suggestions)
        if suggest_str:
            raise UnitError(f"Unrecognized unit: {unit_str}. {suggest_str}")
    
    raise UnitError(f"Unrecognized unit: {unit_str}")


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value from one unit to another.
    
    Args:
        value: The value to convert
        from_unit: The source unit
        to_unit: The target unit
        
    Returns:
        Converted value
        
    Raises:
        UnitError: If the conversion is not possible
    """
    # Special case for temperature conversions
    if from_unit in ('C', 'F') and to_unit in ('C', 'F', 'K'):
        # First convert to Kelvin
        if from_unit == 'C':
            kelvin = value + 273.15
        else:  # from_unit == 'F'
            kelvin = (value - 32.0) * 5.0/9.0 + 273.15
        
        # Then convert from Kelvin to target
        if to_unit == 'K':
            return kelvin
        elif to_unit == 'C':
            return kelvin - 273.15
        else:  # to_unit == 'F'
            return (kelvin - 273.15) * 9.0/5.0 + 32.0
    
    # Normal unit conversion
    from_factor, from_base = parse_unit(from_unit)
    to_factor, to_base = parse_unit(to_unit)
    
    # Check if base units are compatible
    if from_base != to_base and from_base != 'dimensionless' and to_base != 'dimensionless':
        # Special case for frequency and angular frequency conversions
        if (from_unit in FREQUENCY_UNITS and to_unit in FREQUENCY_UNITS and
            FREQUENCY_UNITS[from_unit] != FREQUENCY_UNITS[to_unit]):
            # This is a conversion between frequency and angular frequency,
            # which is handled by the unit factors
            pass
        else:
            raise UnitError(f"Incompatible units: {from_unit} and {to_unit}")
    
    # Convert value
    return value * from_factor / to_factor


def validate_units(expected_unit: str, required: bool = True):
    """Decorator to validate units on function parameters and return values.
    
    Args:
        expected_unit: The expected unit for the decorated function's value
        required: Whether the unit is required (if False, no validation is done
                 when the unit is not provided)
    
    Returns:
        Decorated function that validates units
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Check if 'unit' is in kwargs
            input_unit = kwargs.get('unit')
            
            # If unit is required but not provided, raise error
            if required and input_unit is None:
                raise UnitError(f"Function {func.__name__} requires a unit parameter")
            
            # If unit is provided but doesn't match expected, convert
            if input_unit is not None and input_unit != expected_unit:
                # Verify the input unit is recognized
                if input_unit not in UNIT_FACTORS and not (
                    '/' in input_unit and 
                    all(part in UNIT_FACTORS for part in input_unit.split('/'))
                ):
                    raise UnitError(f"Unrecognized unit: {input_unit}")
                    
                # Find parameters that need conversion
                for param_name, param_value in list(kwargs.items()):
                    # Skip the unit parameter itself and non-numeric parameters
                    if param_name == 'unit' or not isinstance(param_value, (int, float)):
                        continue
                    
                    # Convert parameter value
                    try:
                        kwargs[param_name] = convert_value(
                            param_value, input_unit, expected_unit
                        )
                        logger.debug(
                            f"Converted {param_name} from {input_unit} to {expected_unit}: "
                            f"{param_value} -> {kwargs[param_name]}"
                        )
                    except UnitError as e:
                        raise UnitError(
                            f"Parameter {param_name} has incompatible unit {input_unit}, "
                            f"expected {expected_unit}"
                        ) from e
                
                # Update unit to expected
                kwargs['unit'] = expected_unit
            
            # Call original function
            result = func(*args, **kwargs)
            
            # Add unit info to result if it's a dict and doesn't already have it
            if isinstance(result, dict) and 'unit' not in result:
                result = result.copy()  # Don't modify the original
                result['unit'] = expected_unit
            
            return result
        
        return wrapper
    
    return decorator


def get_unit_type(unit_str: str) -> str:
    """Get the type of a unit (e.g., 'frequency', 'angular_frequency').
    
    Args:
        unit_str: The unit string (e.g., 'Hz', 'rad/s')
        
    Returns:
        Unit type as a string
    """
    if unit_str in FREQUENCY_UNITS:
        return FREQUENCY_UNITS[unit_str]
    
    if unit_str in UNIT_DIMENSIONS:
        # Determine type from dimensions
        dimensions = UNIT_DIMENSIONS[unit_str]
        
        if all(d == 0 for d in dimensions):
            return 'dimensionless'
        
        if dimensions[0] == 1 and all(d == 0 for i, d in enumerate(dimensions) if i != 0):
            return 'angle'
        
        if dimensions[1] == 1 and all(d == 0 for i, d in enumerate(dimensions) if i != 1):
            return 'time'
        
        if dimensions[1] == -1 and all(d == 0 for i, d in enumerate(dimensions) if i != 1):
            return 'frequency'
        
        if dimensions[2] == 1 and all(d == 0 for i, d in enumerate(dimensions) if i != 2):
            return 'length'
        
        # Add more type checks as needed
    
    # If we can't determine the type, use unit as the type
    return unit_str


# Example usage
if __name__ == "__main__":
    # Example function using the decorator
    @validate_units(expected_unit='rad')
    def compute_angle(theta, unit='rad'):
        """Compute something with an angle."""
        # Function will receive theta in radians, regardless of input unit
        return {'angle': theta, 'sin': math.sin(theta)}
    
    # Test with different units
    result1 = compute_angle(theta=math.pi/4, unit='rad')  # No conversion needed
    result2 = compute_angle(theta=45, unit='deg')         # Converts 45° to π/4 rad
    result3 = compute_angle(theta=50, unit='grad')        # Converts 50 grad to π/4 rad
    
    print(f"Result with radians: {result1}")
    print(f"Result with degrees: {result2}")
    print(f"Result with gradians: {result3}")
    
    # All results should be the same
    assert abs(result1['angle'] - result2['angle']) < 1e-10
    assert abs(result1['sin'] - result2['sin']) < 1e-10
    assert abs(result1['angle'] - result3['angle']) < 1e-10
    assert abs(result1['sin'] - result3['sin']) < 1e-10
