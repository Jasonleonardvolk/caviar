"""
ELFIN Template System

This module provides template definition and instantiation capabilities for ELFIN.
Templates are parametrized components that can be instantiated with specific values.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class TemplateParameter:
    """Represents a parameter in a template definition."""
    name: str
    default_value: Any = None
    parameter_type: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of a template parameter."""
        result = self.name
        if self.parameter_type:
            result += f": {self.parameter_type}"
        if self.default_value is not None:
            result += f" = {self.default_value}"
        return result


@dataclass
class TemplateDefinition:
    """Represents a template definition in ELFIN."""
    name: str
    parameters: List[TemplateParameter] = field(default_factory=list)
    body: Any = None
    source: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of a template definition."""
        params_str = ", ".join(str(p) for p in self.parameters)
        return f"template {self.name}({params_str})"
    
    def get_parameter(self, name: str) -> Optional[TemplateParameter]:
        """Get a parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def get_parameter_names(self) -> List[str]:
        """Get a list of parameter names."""
        return [param.name for param in self.parameters]
    
    def has_default_values(self) -> bool:
        """Check if all parameters have default values."""
        return all(param.default_value is not None for param in self.parameters)
    
    def get_required_parameters(self) -> List[TemplateParameter]:
        """Get a list of parameters without default values."""
        return [param for param in self.parameters if param.default_value is None]


@dataclass
class TemplateArgument:
    """Represents an argument in a template instantiation."""
    value: Any
    name: Optional[str] = None
    
    @property
    def is_named(self) -> bool:
        """Check if this is a named argument."""
        return self.name is not None


@dataclass
class TemplateInstance:
    """Represents an instantiation of a template with specific arguments."""
    template_def: TemplateDefinition
    arguments: List[TemplateArgument] = field(default_factory=list)
    instance_name: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of a template instance."""
        args_str = ", ".join(
            f"{arg.name}={arg.value}" if arg.is_named else str(arg.value)
            for arg in self.arguments
        )
        if self.instance_name:
            return f"{self.instance_name}: {self.template_def.name}({args_str})"
        return f"{self.template_def.name}({args_str})"
    
    def get_argument_by_name(self, name: str) -> Optional[TemplateArgument]:
        """Get an argument by parameter name."""
        for arg in self.arguments:
            if arg.is_named and arg.name == name:
                return arg
        return None
    
    def get_argument_by_position(self, position: int) -> Optional[TemplateArgument]:
        """Get an argument by position."""
        positional_args = [arg for arg in self.arguments if not arg.is_named]
        if 0 <= position < len(positional_args):
            return positional_args[position]
        return None


class TemplateRegistry:
    """
    Registry for template definitions.
    
    This class keeps track of template definitions and provides methods for
    instantiating templates with specific arguments.
    """
    
    def __init__(self):
        """Initialize an empty template registry."""
        self.templates: Dict[str, TemplateDefinition] = {}
    
    def register(self, template: TemplateDefinition) -> None:
        """
        Register a template definition.
        
        Args:
            template: The template to register
            
        Raises:
            ValueError: If a template with the same name is already registered
        """
        if template.name in self.templates:
            raise ValueError(f"Template '{template.name}' already registered")
        
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[TemplateDefinition]:
        """
        Get a template by name.
        
        Args:
            name: The name of the template
            
        Returns:
            The template if found, None otherwise
        """
        return self.templates.get(name)
    
    def instantiate(self, template_name: str, args: List[TemplateArgument],
                   instance_name: Optional[str] = None) -> TemplateInstance:
        """
        Instantiate a template with specific arguments.
        
        Args:
            template_name: The name of the template to instantiate
            args: The arguments to pass to the template
            instance_name: The name of the instance (optional)
            
        Returns:
            The template instance
            
        Raises:
            ValueError: If the template is not found or if the arguments are invalid
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Validate arguments
        self._validate_arguments(template, args)
        
        # Create the template instance
        return TemplateInstance(template, args, instance_name)
    
    def _validate_arguments(self, template: TemplateDefinition,
                           args: List[TemplateArgument]) -> None:
        """
        Validate that the arguments match the template parameters.
        
        Args:
            template: The template to validate against
            args: The arguments to validate
            
        Raises:
            ValueError: If the arguments are invalid
        """
        # Check for named arguments that don't match any parameter
        for arg in args:
            if arg.is_named and not template.get_parameter(arg.name):
                raise ValueError(f"Unknown parameter '{arg.name}' in template '{template.name}'")
        
        # Check that all required parameters are provided
        required_params = template.get_required_parameters()
        provided_params = set()
        
        # Add positional arguments
        positional_args = [arg for arg in args if not arg.is_named]
        param_names = template.get_parameter_names()
        
        for i, arg in enumerate(positional_args):
            if i < len(param_names):
                provided_params.add(param_names[i])
            else:
                raise ValueError(f"Too many positional arguments for template '{template.name}'")
        
        # Add named arguments
        for arg in args:
            if arg.is_named:
                provided_params.add(arg.name)
        
        # Check for missing required parameters
        missing_params = [param.name for param in required_params
                         if param.name not in provided_params]
        
        if missing_params:
            params_str = ", ".join(missing_params)
            raise ValueError(f"Missing required parameters in template '{template.name}': {params_str}")


def parse_template_declaration(declaration: str) -> TemplateDefinition:
    """
    Parse a template declaration string.
    
    Args:
        declaration: The template declaration string
        
    Returns:
        A TemplateDefinition object
        
    Example:
        >>> parse_template_declaration("template Vector3(x=0.0, y=0.0, z=0.0)")
        TemplateDefinition(name='Vector3', parameters=[
            TemplateParameter(name='x', default_value=0.0),
            TemplateParameter(name='y', default_value=0.0),
            TemplateParameter(name='z', default_value=0.0)
        ])
    """
    # This is a placeholder implementation - in a real parser we would use
    # a proper parsing library. For now, we'll use a simple regex-based
    # approach for demonstration purposes.
    import re
    
    # Match template name and parameters
    template_pattern = r'template\s+(\w+)\s*\(\s*(.*?)\s*\)'
    match = re.match(template_pattern, declaration)
    
    if not match:
        raise ValueError(f"Invalid template declaration: {declaration}")
    
    name, params_str = match.groups()
    
    # Parse parameters
    parameters = []
    if params_str:
        for param_str in re.split(r'\s*,\s*', params_str):
            # Check for parameter with default value
            default_match = re.match(r'(\w+)(?:\s*:\s*(\w+))?\s*=\s*(.+)', param_str)
            if default_match:
                param_name, param_type, default_value = default_match.groups()
                # In a real implementation, we would parse the default value
                # based on the parameter type. For now, we'll just use it as a string.
                parameters.append(TemplateParameter(param_name, default_value, param_type))
            else:
                # Check for parameter with type but no default
                type_match = re.match(r'(\w+)\s*:\s*(\w+)', param_str)
                if type_match:
                    param_name, param_type = type_match.groups()
                    parameters.append(TemplateParameter(param_name, None, param_type))
                else:
                    # Simple parameter with no type or default
                    parameters.append(TemplateParameter(param_str))
    
    return TemplateDefinition(name=name, parameters=parameters)


def parse_template_instantiation(instantiation: str) -> Tuple[str, List[TemplateArgument]]:
    """
    Parse a template instantiation string.
    
    Args:
        instantiation: The template instantiation string
        
    Returns:
        A tuple of (template_name, arguments)
        
    Example:
        >>> parse_template_instantiation("Vector3(1.0, y=2.0, z=3.0)")
        ('Vector3', [
            TemplateArgument(value=1.0),
            TemplateArgument(value=2.0, name='y'),
            TemplateArgument(value=3.0, name='z')
        ])
    """
    # This is a placeholder implementation - in a real parser we would use
    # a proper parsing library. For now, we'll use a simple regex-based
    # approach for demonstration purposes.
    import re
    
    # Match template name and arguments
    instantiation_pattern = r'(\w+)\s*\(\s*(.*?)\s*\)'
    match = re.match(instantiation_pattern, instantiation)
    
    if not match:
        raise ValueError(f"Invalid template instantiation: {instantiation}")
    
    name, args_str = match.groups()
    
    # Parse arguments
    arguments = []
    if args_str:
        for arg_str in re.split(r'\s*,\s*', args_str):
            # Check for named argument
            named_match = re.match(r'(\w+)\s*=\s*(.+)', arg_str)
            if named_match:
                arg_name, arg_value = named_match.groups()
                # In a real implementation, we would parse the argument value
                # based on the parameter type. For now, we'll just use it as a string.
                arguments.append(TemplateArgument(arg_value, arg_name))
            else:
                # Positional argument
                arguments.append(TemplateArgument(arg_str))
    
    return name, arguments
