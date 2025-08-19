"""
ELFIN Module AST (Abstract Syntax Tree) Extensions.

This module defines additional AST nodes for the ELFIN module system,
including imports and templates.
"""

from typing import List, Dict, Optional, Any, Union, Tuple
from alan_backend.elfin.parser.ast import Node, Expression


class ImportDecl(Node):
    """Import declaration node."""
    
    def __init__(
        self,
        imports: List[Dict[str, str]],
        source: str,
        location: Tuple[int, int] = (0, 0)
    ):
        """
        Initialize an import declaration.
        
        Args:
            imports: List of imports, each with 'name' and 'alias' keys
            source: The source module path
            location: The (line, column) location of the import
        """
        super().__init__()
        self.imports = imports
        self.source = source
        self.line, self.column = location


class TemplateParamDecl(Node):
    """Template parameter declaration node."""
    
    def __init__(
        self,
        name: str,
        param_type: Optional[str] = None,
        default_value: Optional[Any] = None,
        location: Tuple[int, int] = (0, 0)
    ):
        """
        Initialize a template parameter declaration.
        
        Args:
            name: The parameter name
            param_type: The parameter type (optional)
            default_value: The default value (optional)
            location: The (line, column) location of the parameter
        """
        super().__init__()
        self.name = name
        self.param_type = param_type
        self.default_value = default_value
        self.line, self.column = location


class TemplateDecl(Node):
    """Template declaration node."""
    
    def __init__(
        self,
        name: str,
        parameters: List[TemplateParamDecl],
        body: Node,
        location: Tuple[int, int] = (0, 0)
    ):
        """
        Initialize a template declaration.
        
        Args:
            name: The template name
            parameters: List of parameter declarations
            body: The template body
            location: The (line, column) location of the template
        """
        super().__init__()
        self.name = name
        self.parameters = parameters
        self.body = body
        self.line, self.column = location
        self.add_child(body)


class TemplateArgument(Node):
    """Template argument node."""
    
    def __init__(
        self,
        value: Any,
        name: Optional[str] = None,
        location: Tuple[int, int] = (0, 0)
    ):
        """
        Initialize a template argument.
        
        Args:
            value: The argument value
            name: The argument name (optional)
            location: The (line, column) location of the argument
        """
        super().__init__()
        self.value = value
        self.name = name
        self.line, self.column = location


class TemplateInstantiation(Expression):
    """Template instantiation node."""
    
    def __init__(
        self,
        template_name: str,
        arguments: List[TemplateArgument],
        instance_name: Optional[str] = None,
        location: Tuple[int, int] = (0, 0)
    ):
        """
        Initialize a template instantiation.
        
        Args:
            template_name: The name of the template
            arguments: List of arguments
            instance_name: The name of the instance (optional)
            location: The (line, column) location of the instantiation
        """
        super().__init__()
        self.template_name = template_name
        self.arguments = arguments
        self.instance_name = instance_name
        self.line, self.column = location
        for arg in arguments:
            self.add_child(arg)


class ModuleNode(Node):
    """Module node representing a complete ELFIN module."""
    
    def __init__(
        self,
        path: str,
        declarations: List[Node] = None,
        imports: List[ImportDecl] = None,
        templates: List[TemplateDecl] = None
    ):
        """
        Initialize a module.
        
        Args:
            path: The path to the module file
            declarations: List of declarations (optional)
            imports: List of imports (optional)
            templates: List of templates (optional)
        """
        super().__init__()
        self.path = path
        self.declarations = declarations or []
        self.imports = imports or []
        self.templates = templates or []
        
        # Add children
        for decl in self.declarations:
            self.add_child(decl)
        for imp in self.imports:
            self.add_child(imp)
        for tmpl in self.templates:
            self.add_child(tmpl)
