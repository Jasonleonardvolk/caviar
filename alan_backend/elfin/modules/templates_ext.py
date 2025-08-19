"""
ELFIN Template Extensions.

This module provides extended template functionality for ELFIN, including
template specialization, template inheritance, and advanced template features.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from alan_backend.elfin.modules.templates import TemplateRegistry, TemplateDefinition, TemplateParameter


class TemplateSpecialization:
    """
    A template specialization.
    
    This class represents a specialized version of a template for specific parameter values.
    """
    
    def __init__(
        self,
        base_template: str,
        parameters: Dict[str, Any],
        body: Any,
        source: Optional[str] = None
    ):
        """
        Initialize a template specialization.
        
        Args:
            base_template: The name of the base template
            parameters: Dictionary of parameter values for this specialization
            body: The specialized template body
            source: The source of the specialization (optional)
        """
        self.base_template = base_template
        self.parameters = parameters
        self.body = body
        self.source = source
    
    def matches(self, parameters: Dict[str, Any]) -> bool:
        """
        Check if this specialization matches the given parameters.
        
        Args:
            parameters: Dictionary of parameter values to check
            
        Returns:
            True if this specialization matches the parameters, False otherwise
        """
        # Check if all specialization parameters are matched
        for name, value in self.parameters.items():
            if name not in parameters or parameters[name] != value:
                return False
        
        return True


class TemplateInheritance:
    """
    A template inheritance relationship.
    
    This class represents a template that inherits from another template.
    """
    
    def __init__(
        self,
        child_template: str,
        parent_template: str,
        parameters: Dict[str, Any],
        source: Optional[str] = None
    ):
        """
        Initialize a template inheritance relationship.
        
        Args:
            child_template: The name of the child template
            parent_template: The name of the parent template
            parameters: Dictionary of parameter values to pass to the parent
            source: The source of the inheritance (optional)
        """
        self.child_template = child_template
        self.parent_template = parent_template
        self.parameters = parameters
        self.source = source


class ExtendedTemplateRegistry(TemplateRegistry):
    """
    An extended template registry with support for specialization and inheritance.
    
    This registry extends the base TemplateRegistry with support for template
    specialization, inheritance, and advanced template features.
    """
    
    def __init__(self):
        """Initialize an empty extended template registry."""
        super().__init__()
        self.specializations: Dict[str, List[TemplateSpecialization]] = {}
        self.inheritance: Dict[str, TemplateInheritance] = {}
    
    def register_specialization(
        self,
        base_template: str,
        parameters: Dict[str, Any],
        body: Any,
        source: Optional[str] = None
    ) -> None:
        """
        Register a template specialization.
        
        Args:
            base_template: The name of the base template
            parameters: Dictionary of parameter values for this specialization
            body: The specialized template body
            source: The source of the specialization (optional)
            
        Raises:
            ValueError: If the base template is not registered
        """
        # Check if the base template exists
        if base_template not in self.templates:
            raise ValueError(f"Base template '{base_template}' not found")
        
        # Create the specialization
        specialization = TemplateSpecialization(
            base_template=base_template,
            parameters=parameters,
            body=body,
            source=source
        )
        
        # Add to the specializations dictionary
        if base_template not in self.specializations:
            self.specializations[base_template] = []
        
        self.specializations[base_template].append(specialization)
    
    def register_inheritance(
        self,
        child_template: str,
        parent_template: str,
        parameters: Dict[str, Any],
        source: Optional[str] = None
    ) -> None:
        """
        Register a template inheritance relationship.
        
        Args:
            child_template: The name of the child template
            parent_template: The name of the parent template
            parameters: Dictionary of parameter values to pass to the parent
            source: The source of the inheritance (optional)
            
        Raises:
            ValueError: If the parent template is not registered or if the child template is already registered
        """
        # Check if the parent template exists
        if parent_template not in self.templates:
            raise ValueError(f"Parent template '{parent_template}' not found")
        
        # Check if the child template is already registered
        if child_template in self.templates:
            raise ValueError(f"Child template '{child_template}' already registered")
        
        # Create the inheritance relationship
        inheritance = TemplateInheritance(
            child_template=child_template,
            parent_template=parent_template,
            parameters=parameters,
            source=source
        )
        
        # Add to the inheritance dictionary
        self.inheritance[child_template] = inheritance
    
    def get_specialization(
        self,
        template_name: str,
        parameters: Dict[str, Any]
    ) -> Optional[TemplateSpecialization]:
        """
        Get a template specialization for specific parameter values.
        
        Args:
            template_name: The name of the template
            parameters: Dictionary of parameter values
            
        Returns:
            The template specialization if found, None otherwise
        """
        # Check if there are specializations for this template
        if template_name not in self.specializations:
            return None
        
        # Check each specialization
        for specialization in self.specializations[template_name]:
            if specialization.matches(parameters):
                return specialization
        
        return None
    
    def get_parent_template(
        self,
        template_name: str
    ) -> Optional[Tuple[TemplateDefinition, Dict[str, Any]]]:
        """
        Get the parent template of a template.
        
        Args:
            template_name: The name of the template
            
        Returns:
            A tuple of (parent_template, parameters) if found, None otherwise
        """
        # Check if there is an inheritance relationship for this template
        if template_name not in self.inheritance:
            return None
        
        # Get the inheritance relationship
        inheritance = self.inheritance[template_name]
        
        # Get the parent template
        parent_template = self.get_template(inheritance.parent_template)
        if parent_template is None:
            return None
        
        return parent_template, inheritance.parameters
    
    def instantiate(
        self,
        template_name: str,
        parameters: Dict[str, Any],
        instance_name: Optional[str] = None
    ) -> Any:
        """
        Instantiate a template with specific parameters.
        
        This method handles specialization and inheritance to instantiate the
        most specific version of a template.
        
        Args:
            template_name: The name of the template
            parameters: Dictionary of parameter values
            instance_name: The name of the instance (optional)
            
        Returns:
            The instantiated template
            
        Raises:
            ValueError: If the template is not found or if the parameters are invalid
        """
        # Check for a specialization first
        specialization = self.get_specialization(template_name, parameters)
        if specialization:
            # TODO: Instantiate the specialization
            # For now, just return it
            return specialization.body
        
        # Get the template definition
        template_def = self.get_template(template_name)
        if template_def is None:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Validate parameters
        # TODO: Validate parameters
        
        # Get parent template if this is an inheriting template
        parent_result = None
        parent_info = self.get_parent_template(template_name)
        if parent_info:
            parent_template, parent_params = parent_info
            
            # Merge parameters with parent parameters
            merged_params = parent_params.copy()
            merged_params.update(parameters)
            
            # Instantiate the parent template
            parent_result = self.instantiate(
                parent_template.name,
                merged_params
            )
        
        # TODO: Instantiate the template with the parameters
        # For now, just return the template definition and parameters
        return {
            "template": template_def,
            "parameters": parameters,
            "instance_name": instance_name,
            "parent_result": parent_result
        }
