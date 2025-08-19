"""
ELFIN Conditional Compilation.

This module provides conditional compilation features for ELFIN templates,
allowing templates to include or exclude sections based on parameters.
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from enum import Enum, auto


class ConditionType(Enum):
    """Types of conditions supported in conditional compilation."""
    
    EQUALS = auto()
    NOT_EQUALS = auto()
    LESS_THAN = auto()
    LESS_EQUALS = auto()
    GREATER_THAN = auto()
    GREATER_EQUALS = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IS_TYPE = auto()
    HAS_PROPERTY = auto()
    DEFINED = auto()


class Condition:
    """
    A condition for conditional compilation.
    
    Conditions can be evaluated against a set of parameters to determine
    whether a section of code should be included or excluded.
    """
    
    def __init__(self, condition_type: ConditionType, *args):
        """
        Initialize a condition.
        
        Args:
            condition_type: The type of condition
            *args: Arguments for the condition (depends on the type)
        """
        self.condition_type = condition_type
        self.args = args
    
    def evaluate(self, parameters: Dict[str, Any]) -> bool:
        """
        Evaluate the condition against a set of parameters.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            True if the condition is satisfied, False otherwise
        """
        if self.condition_type == ConditionType.EQUALS:
            param_name, value = self.args
            return param_name in parameters and parameters[param_name] == value
            
        elif self.condition_type == ConditionType.NOT_EQUALS:
            param_name, value = self.args
            return param_name in parameters and parameters[param_name] != value
            
        elif self.condition_type == ConditionType.LESS_THAN:
            param_name, value = self.args
            return param_name in parameters and parameters[param_name] < value
            
        elif self.condition_type == ConditionType.LESS_EQUALS:
            param_name, value = self.args
            return param_name in parameters and parameters[param_name] <= value
            
        elif self.condition_type == ConditionType.GREATER_THAN:
            param_name, value = self.args
            return param_name in parameters and parameters[param_name] > value
            
        elif self.condition_type == ConditionType.GREATER_EQUALS:
            param_name, value = self.args
            return param_name in parameters and parameters[param_name] >= value
            
        elif self.condition_type == ConditionType.AND:
            subconditions = self.args
            return all(subcond.evaluate(parameters) for subcond in subconditions)
            
        elif self.condition_type == ConditionType.OR:
            subconditions = self.args
            return any(subcond.evaluate(parameters) for subcond in subconditions)
            
        elif self.condition_type == ConditionType.NOT:
            subcondition = self.args[0]
            return not subcondition.evaluate(parameters)
            
        elif self.condition_type == ConditionType.IS_TYPE:
            param_name, type_name = self.args
            if param_name not in parameters:
                return False
                
            # Check the type of the parameter
            param_value = parameters[param_name]
            
            # Basic type checking
            if type_name == "int":
                return isinstance(param_value, int)
            elif type_name == "float":
                return isinstance(param_value, (float, int))
            elif type_name == "string":
                return isinstance(param_value, str)
            elif type_name == "bool":
                return isinstance(param_value, bool)
            elif type_name == "list":
                return isinstance(param_value, list)
            elif type_name == "dict":
                return isinstance(param_value, dict)
            else:
                # Assume it's a user-defined type
                # This would need to be integrated with the actual type system
                return hasattr(param_value, "__type__") and param_value.__type__ == type_name
                
        elif self.condition_type == ConditionType.HAS_PROPERTY:
            param_name, property_name = self.args
            if param_name not in parameters:
                return False
                
            param_value = parameters[param_name]
            return hasattr(param_value, property_name)
            
        elif self.condition_type == ConditionType.DEFINED:
            param_name = self.args[0]
            return param_name in parameters
            
        # Default case
        return False
    
    @staticmethod
    def equals(param_name: str, value: Any) -> 'Condition':
        """Create an equals condition."""
        return Condition(ConditionType.EQUALS, param_name, value)
    
    @staticmethod
    def not_equals(param_name: str, value: Any) -> 'Condition':
        """Create a not equals condition."""
        return Condition(ConditionType.NOT_EQUALS, param_name, value)
    
    @staticmethod
    def less_than(param_name: str, value: Any) -> 'Condition':
        """Create a less than condition."""
        return Condition(ConditionType.LESS_THAN, param_name, value)
    
    @staticmethod
    def less_equals(param_name: str, value: Any) -> 'Condition':
        """Create a less equals condition."""
        return Condition(ConditionType.LESS_EQUALS, param_name, value)
    
    @staticmethod
    def greater_than(param_name: str, value: Any) -> 'Condition':
        """Create a greater than condition."""
        return Condition(ConditionType.GREATER_THAN, param_name, value)
    
    @staticmethod
    def greater_equals(param_name: str, value: Any) -> 'Condition':
        """Create a greater equals condition."""
        return Condition(ConditionType.GREATER_EQUALS, param_name, value)
    
    @staticmethod
    def and_(*conditions) -> 'Condition':
        """Create an AND condition."""
        return Condition(ConditionType.AND, *conditions)
    
    @staticmethod
    def or_(*conditions) -> 'Condition':
        """Create an OR condition."""
        return Condition(ConditionType.OR, *conditions)
    
    @staticmethod
    def not_(condition: 'Condition') -> 'Condition':
        """Create a NOT condition."""
        return Condition(ConditionType.NOT, condition)
    
    @staticmethod
    def is_type(param_name: str, type_name: str) -> 'Condition':
        """Create an IS_TYPE condition."""
        return Condition(ConditionType.IS_TYPE, param_name, type_name)
    
    @staticmethod
    def has_property(param_name: str, property_name: str) -> 'Condition':
        """Create a HAS_PROPERTY condition."""
        return Condition(ConditionType.HAS_PROPERTY, param_name, property_name)
    
    @staticmethod
    def defined(param_name: str) -> 'Condition':
        """Create a DEFINED condition."""
        return Condition(ConditionType.DEFINED, param_name)


class ConditionalSection:
    """
    A conditional section in a template.
    
    A conditional section contains code that is included or excluded
    based on a condition.
    """
    
    def __init__(self, condition: Condition, body: Any, else_body: Optional[Any] = None):
        """
        Initialize a conditional section.
        
        Args:
            condition: The condition for this section
            body: The body to include if the condition is true
            else_body: The body to include if the condition is false (optional)
        """
        self.condition = condition
        self.body = body
        self.else_body = else_body
    
    def evaluate(self, parameters: Dict[str, Any]) -> Any:
        """
        Evaluate the conditional section against a set of parameters.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            The body if the condition is true, the else_body if it's false
        """
        if self.condition.evaluate(parameters):
            return self.body
        elif self.else_body is not None:
            return self.else_body
        else:
            return None


class ConditionalCompiler:
    """
    A compiler for conditional sections.
    
    This compiler processes template bodies with conditional sections,
    including or excluding sections based on parameter values.
    """
    
    def __init__(self):
        """Initialize the conditional compiler."""
        pass
    
    def process_template(self, template_body: Any, parameters: Dict[str, Any]) -> Any:
        """
        Process a template body with conditional sections.
        
        Args:
            template_body: The template body to process
            parameters: Dictionary of parameter values
            
        Returns:
            The processed template body
        """
        # In a real implementation, the template body would be an AST
        # with conditional sections. For now, we'll just return it.
        return template_body
    
    def process_conditional_section(self, section: ConditionalSection, parameters: Dict[str, Any]) -> Any:
        """
        Process a conditional section.
        
        Args:
            section: The conditional section to process
            parameters: Dictionary of parameter values
            
        Returns:
            The processed section
        """
        return section.evaluate(parameters)


# DSL-like helper functions for creating conditions
def eq(param_name: str, value: Any) -> Condition:
    """Create an equals condition."""
    return Condition.equals(param_name, value)

def ne(param_name: str, value: Any) -> Condition:
    """Create a not equals condition."""
    return Condition.not_equals(param_name, value)

def lt(param_name: str, value: Any) -> Condition:
    """Create a less than condition."""
    return Condition.less_than(param_name, value)

def le(param_name: str, value: Any) -> Condition:
    """Create a less equals condition."""
    return Condition.less_equals(param_name, value)

def gt(param_name: str, value: Any) -> Condition:
    """Create a greater than condition."""
    return Condition.greater_than(param_name, value)

def ge(param_name: str, value: Any) -> Condition:
    """Create a greater equals condition."""
    return Condition.greater_equals(param_name, value)

def and_(*conditions) -> Condition:
    """Create an AND condition."""
    return Condition.and_(*conditions)

def or_(*conditions) -> Condition:
    """Create an OR condition."""
    return Condition.or_(*conditions)

def not_(condition: Condition) -> Condition:
    """Create a NOT condition."""
    return Condition.not_(condition)

def is_type(param_name: str, type_name: str) -> Condition:
    """Create an IS_TYPE condition."""
    return Condition.is_type(param_name, type_name)

def has_property(param_name: str, property_name: str) -> Condition:
    """Create a HAS_PROPERTY condition."""
    return Condition.has_property(param_name, property_name)

def defined(param_name: str) -> Condition:
    """Create a DEFINED condition."""
    return Condition.defined(param_name)

def if_(condition: Condition, body: Any, else_body: Optional[Any] = None) -> ConditionalSection:
    """Create a conditional section."""
    return ConditionalSection(condition, body, else_body)
