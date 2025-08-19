"""
Core unit and dimension definitions for ELFIN.

This module defines the basic structures for representing physical dimensions
and units in ELFIN. It provides a Unit class for representing specific physical
units, a UnitDimension class for representing the dimensional signature of a unit,
and a UnitTable for storing and retrieving units.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re


@dataclass
class UnitDimension:
    """
    Represents the dimensional signature of a physical quantity.
    
    Each dimension is represented as a power (exponent) of a base dimension.
    For example, acceleration has dimensions [L]¹[T]⁻².
    """
    # Base dimensions and their exponents
    mass: int = 0          # [M] - kilogram
    length: int = 0        # [L] - meter
    time: int = 0          # [T] - second
    angle: int = 0         # [A] - radian
    temperature: int = 0   # [Θ] - kelvin
    current: int = 0       # [I] - ampere
    luminosity: int = 0    # [J] - candela
    
    @staticmethod
    def dimensionless() -> 'UnitDimension':
        """Create a dimensionless unit dimension."""
        return UnitDimension()
    
    def __mul__(self, other: 'UnitDimension') -> 'UnitDimension':
        """Multiply two unit dimensions (add exponents)."""
        return UnitDimension(
            mass=self.mass + other.mass,
            length=self.length + other.length,
            time=self.time + other.time,
            angle=self.angle + other.angle,
            temperature=self.temperature + other.temperature,
            current=self.current + other.current,
            luminosity=self.luminosity + other.luminosity
        )
    
    def __truediv__(self, other: 'UnitDimension') -> 'UnitDimension':
        """Divide two unit dimensions (subtract exponents)."""
        return UnitDimension(
            mass=self.mass - other.mass,
            length=self.length - other.length,
            time=self.time - other.time,
            angle=self.angle - other.angle,
            temperature=self.temperature - other.temperature,
            current=self.current - other.current,
            luminosity=self.luminosity - other.luminosity
        )
    
    def __pow__(self, exponent: int) -> 'UnitDimension':
        """Raise unit dimension to a power (multiply exponents)."""
        return UnitDimension(
            mass=self.mass * exponent,
            length=self.length * exponent,
            time=self.time * exponent,
            angle=self.angle * exponent,
            temperature=self.temperature * exponent,
            current=self.current * exponent,
            luminosity=self.luminosity * exponent
        )
    
    def __eq__(self, other: object) -> bool:
        """Check if two unit dimensions are equal."""
        if not isinstance(other, UnitDimension):
            return False
        return (
            self.mass == other.mass and
            self.length == other.length and
            self.time == other.time and
            self.angle == other.angle and
            self.temperature == other.temperature and
            self.current == other.current and
            self.luminosity == other.luminosity
        )
    
    def is_dimensionless(self) -> bool:
        """Check if the unit dimension is dimensionless."""
        return (
            self.mass == 0 and
            self.length == 0 and
            self.time == 0 and
            self.angle == 0 and
            self.temperature == 0 and
            self.current == 0 and
            self.luminosity == 0
        )
    
    def is_pure_angle(self) -> bool:
        """Check if the unit dimension represents a pure angle."""
        return (
            self.angle == 1 and
            self.mass == 0 and
            self.length == 0 and
            self.time == 0 and
            self.temperature == 0 and
            self.current == 0 and
            self.luminosity == 0
        )
    
    def __str__(self) -> str:
        """String representation of the unit dimension."""
        parts = []
        dim_names = {
            'mass': 'kg',
            'length': 'm',
            'time': 's',
            'angle': 'rad',
            'temperature': 'K',
            'current': 'A',
            'luminosity': 'cd'
        }
        
        for dim, symbol in dim_names.items():
            exp = getattr(self, dim)
            if exp != 0:
                if exp == 1:
                    parts.append(symbol)
                else:
                    parts.append(f"{symbol}^{exp}")
        
        if not parts:
            return "dimensionless"
        
        return " · ".join(parts)


@dataclass
class Unit:
    """
    Represents a physical unit with a name, symbol, and dimension.
    
    For example, the unit "meter per second" has name="meter_per_second",
    symbol="m/s", and dimensions of [L]¹[T]⁻¹.
    """
    name: str
    symbol: str
    dimension: UnitDimension
    alias: Optional[str] = None
    
    @staticmethod
    def parse(unit_str: str) -> 'Unit':
        """
        Parse a unit string into a Unit object.
        
        This is a very basic parser that handles simple unit strings like "kg",
        "m/s", "N*m", etc. For more complex unit strings, the UnitTable should
        be used to look up predefined units.
        
        Args:
            unit_str: String representation of the unit (e.g., "kg", "m/s")
            
        Returns:
            Unit object representing the parsed unit
        """
        # Simple case: single unit
        if unit_str in BASE_UNITS:
            return BASE_UNITS[unit_str]
        
        # Try to find in the default unit table
        unit_table = UnitTable()
        if unit_str in unit_table.units:
            return unit_table.units[unit_str]
        
        # Not found, could implement more complex parsing here
        raise ValueError(f"Unknown unit: {unit_str}")
    
    def __mul__(self, other: 'Unit') -> 'Unit':
        """Multiply two units."""
        name = f"{self.name}_{other.name}"
        symbol = f"{self.symbol}·{other.symbol}"
        dimension = self.dimension * other.dimension
        return Unit(name, symbol, dimension)
    
    def __truediv__(self, other: 'Unit') -> 'Unit':
        """Divide two units."""
        name = f"{self.name}_per_{other.name}"
        symbol = f"{self.symbol}/{other.symbol}"
        dimension = self.dimension / other.dimension
        return Unit(name, symbol, dimension)
    
    def __pow__(self, exponent: int) -> 'Unit':
        """Raise a unit to a power."""
        name = f"{self.name}_pow_{exponent}"
        if exponent == 2:
            symbol = f"{self.symbol}²"
        elif exponent == 3:
            symbol = f"{self.symbol}³"
        else:
            symbol = f"{self.symbol}^{exponent}"
        dimension = self.dimension ** exponent
        return Unit(name, symbol, dimension)
    
    def __eq__(self, other: object) -> bool:
        """Check if two units are equal (same dimension)."""
        if not isinstance(other, Unit):
            return False
        return self.dimension == other.dimension
    
    def __str__(self) -> str:
        """String representation of the unit."""
        return self.symbol


class UnitTable:
    """
    Table of physical units and their dimensions.
    
    This class provides a registry of common physical units and methods for
    looking up units by name, symbol, or dimension.
    """
    
    def __init__(self):
        """Initialize the unit table with common physical units."""
        self.units: Dict[str, Unit] = {}
        
        # Add base units
        for unit in BASE_UNITS.values():
            self.add_unit(unit)
        
        # Add derived units
        self._add_derived_units()
    
    def add_unit(self, unit: Unit) -> None:
        """Add a unit to the table."""
        self.units[unit.symbol] = unit
        if unit.alias:
            self.units[unit.alias] = unit
    
    def get_unit(self, unit_str: str) -> Optional[Unit]:
        """Get a unit by symbol or name."""
        return self.units.get(unit_str)
    
    def get_unit_by_dimension(self, dimension: UnitDimension) -> Optional[Unit]:
        """Get a unit by dimension (returns the first match)."""
        for unit in self.units.values():
            if unit.dimension == dimension:
                return unit
        return None
    
    def _add_derived_units(self) -> None:
        """Add common derived units to the table."""
        # Force units
        newton = Unit(
            name="newton",
            symbol="N",
            dimension=UnitDimension(mass=1, length=1, time=-2),
            alias="force"
        )
        self.add_unit(newton)
        
        # Energy units
        joule = Unit(
            name="joule",
            symbol="J",
            dimension=UnitDimension(mass=1, length=2, time=-2),
            alias="energy"
        )
        self.add_unit(joule)
        
        # Power units
        watt = Unit(
            name="watt",
            symbol="W",
            dimension=UnitDimension(mass=1, length=2, time=-3),
            alias="power"
        )
        self.add_unit(watt)
        
        # Torque units
        newton_meter = Unit(
            name="newton_meter",
            symbol="N·m",
            dimension=UnitDimension(mass=1, length=2, time=-2),
            alias="torque"
        )
        self.add_unit(newton_meter)
        
        # Angular momentum units
        angular_momentum = Unit(
            name="angular_momentum",
            symbol="kg·m²/s",
            dimension=UnitDimension(mass=1, length=2, time=-1),
        )
        self.add_unit(angular_momentum)
        
        # Velocity units
        meter_per_second = Unit(
            name="meter_per_second",
            symbol="m/s",
            dimension=UnitDimension(length=1, time=-1),
            alias="velocity"
        )
        self.add_unit(meter_per_second)
        
        # Acceleration units
        meter_per_second_squared = Unit(
            name="meter_per_second_squared",
            symbol="m/s²",
            dimension=UnitDimension(length=1, time=-2),
            alias="acceleration"
        )
        self.add_unit(meter_per_second_squared)
        
        # Angular velocity units
        radian_per_second = Unit(
            name="radian_per_second",
            symbol="rad/s",
            dimension=UnitDimension(angle=1, time=-1),
            alias="angular_velocity"
        )
        self.add_unit(radian_per_second)
        
        # Angular acceleration units
        radian_per_second_squared = Unit(
            name="radian_per_second_squared",
            symbol="rad/s²",
            dimension=UnitDimension(angle=1, time=-2),
            alias="angular_acceleration"
        )
        self.add_unit(radian_per_second_squared)
        
        # Rotational damping units (torque per angular velocity)
        # Corrected to use N·m·s/rad as per feedback
        rotational_damping = Unit(
            name="rotational_damping",
            symbol="N·m·s/rad",
            dimension=UnitDimension(mass=1, length=2, time=-1, angle=-1),
            alias="angular_damping"
        )
        self.add_unit(rotational_damping)


# Define base units
BASE_UNITS = {
    "kg": Unit("kilogram", "kg", UnitDimension(mass=1)),
    "m": Unit("meter", "m", UnitDimension(length=1)),
    "s": Unit("second", "s", UnitDimension(time=1)),
    "rad": Unit("radian", "rad", UnitDimension(angle=1)),
    "K": Unit("kelvin", "K", UnitDimension(temperature=1)),
    "A": Unit("ampere", "A", UnitDimension(current=1)),
    "cd": Unit("candela", "cd", UnitDimension(luminosity=1)),
    "dimensionless": Unit("dimensionless", "", UnitDimension())
}
