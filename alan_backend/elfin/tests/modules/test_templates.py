"""
Tests for the ELFIN template system.

This module tests the functionality of the TemplateRegistry class and related
components that handle template definition and instantiation in ELFIN.
"""

import pytest

from alan_backend.elfin.modules.templates import (
    TemplateParameter,
    TemplateDefinition,
    TemplateArgument,
    TemplateInstance,
    TemplateRegistry,
    parse_template_declaration,
    parse_template_instantiation
)


class TestTemplateParameter:
    """Tests for the TemplateParameter class."""
    
    def test_parameter_creation(self):
        """Test creating a template parameter."""
        # Parameter with name only
        param1 = TemplateParameter("x")
        assert param1.name == "x"
        assert param1.default_value is None
        assert param1.parameter_type is None
        
        # Parameter with name and default value
        param2 = TemplateParameter("y", 42)
        assert param2.name == "y"
        assert param2.default_value == 42
        assert param2.parameter_type is None
        
        # Parameter with name, default value, and type
        param3 = TemplateParameter("z", 3.14, "float")
        assert param3.name == "z"
        assert param3.default_value == 3.14
        assert param3.parameter_type == "float"
    
    def test_parameter_string_representation(self):
        """Test the string representation of a template parameter."""
        # Parameter with name only
        param1 = TemplateParameter("x")
        assert str(param1) == "x"
        
        # Parameter with name and type
        param2 = TemplateParameter("y", parameter_type="int")
        assert str(param2) == "y: int"
        
        # Parameter with name and default value
        param3 = TemplateParameter("z", 42)
        assert str(param3) == "z = 42"
        
        # Parameter with name, type, and default value
        param4 = TemplateParameter("w", 3.14, "float")
        assert str(param4) == "w: float = 3.14"


class TestTemplateDefinition:
    """Tests for the TemplateDefinition class."""
    
    def test_template_creation(self):
        """Test creating a template definition."""
        # Template with no parameters
        template1 = TemplateDefinition("Empty")
        assert template1.name == "Empty"
        assert len(template1.parameters) == 0
        assert template1.body is None
        assert template1.source is None
        
        # Template with parameters
        params = [
            TemplateParameter("x", 1),
            TemplateParameter("y", 2),
            TemplateParameter("z", 3)
        ]
        template2 = TemplateDefinition("Vector3", params, "body", "source.elfin")
        assert template2.name == "Vector3"
        assert len(template2.parameters) == 3
        assert template2.body == "body"
        assert template2.source == "source.elfin"
    
    def test_template_string_representation(self):
        """Test the string representation of a template definition."""
        # Template with no parameters
        template1 = TemplateDefinition("Empty")
        assert str(template1) == "template Empty()"
        
        # Template with parameters
        params = [
            TemplateParameter("x", 1),
            TemplateParameter("y", 2),
            TemplateParameter("z", 3)
        ]
        template2 = TemplateDefinition("Vector3", params)
        assert str(template2) == "template Vector3(x = 1, y = 2, z = 3)"
    
    def test_get_parameter(self):
        """Test getting a parameter by name."""
        params = [
            TemplateParameter("x", 1),
            TemplateParameter("y", 2),
            TemplateParameter("z", 3)
        ]
        template = TemplateDefinition("Vector3", params)
        
        # Get existing parameters
        assert template.get_parameter("x").name == "x"
        assert template.get_parameter("y").name == "y"
        assert template.get_parameter("z").name == "z"
        
        # Get non-existent parameter
        assert template.get_parameter("w") is None
    
    def test_get_parameter_names(self):
        """Test getting a list of parameter names."""
        params = [
            TemplateParameter("x", 1),
            TemplateParameter("y", 2),
            TemplateParameter("z", 3)
        ]
        template = TemplateDefinition("Vector3", params)
        
        assert template.get_parameter_names() == ["x", "y", "z"]
    
    def test_has_default_values(self):
        """Test checking if all parameters have default values."""
        # All parameters have default values
        params1 = [
            TemplateParameter("x", 1),
            TemplateParameter("y", 2),
            TemplateParameter("z", 3)
        ]
        template1 = TemplateDefinition("Vector3", params1)
        assert template1.has_default_values() is True
        
        # Some parameters don't have default values
        params2 = [
            TemplateParameter("x", 1),
            TemplateParameter("y"),
            TemplateParameter("z", 3)
        ]
        template2 = TemplateDefinition("Vector3", params2)
        assert template2.has_default_values() is False
    
    def test_get_required_parameters(self):
        """Test getting a list of parameters without default values."""
        params = [
            TemplateParameter("x", 1),
            TemplateParameter("y"),
            TemplateParameter("z", 3),
            TemplateParameter("w")
        ]
        template = TemplateDefinition("Test", params)
        
        required = template.get_required_parameters()
        assert len(required) == 2
        assert required[0].name == "y"
        assert required[1].name == "w"


class TestTemplateArgument:
    """Tests for the TemplateArgument class."""
    
    def test_argument_creation(self):
        """Test creating a template argument."""
        # Positional argument
        arg1 = TemplateArgument(42)
        assert arg1.value == 42
        assert arg1.name is None
        assert arg1.is_named is False
        
        # Named argument
        arg2 = TemplateArgument(3.14, "pi")
        assert arg2.value == 3.14
        assert arg2.name == "pi"
        assert arg2.is_named is True


class TestTemplateInstance:
    """Tests for the TemplateInstance class."""
    
    def test_instance_creation(self):
        """Test creating a template instance."""
        # Create a template definition
        params = [
            TemplateParameter("x", 1),
            TemplateParameter("y", 2),
            TemplateParameter("z", 3)
        ]
        template = TemplateDefinition("Vector3", params)
        
        # Create arguments
        args = [
            TemplateArgument(10),
            TemplateArgument(20, "y"),
            TemplateArgument(30, "z")
        ]
        
        # Create instance without name
        instance1 = TemplateInstance(template, args)
        assert instance1.template_def is template
        assert len(instance1.arguments) == 3
        assert instance1.instance_name is None
        
        # Create instance with name
        instance2 = TemplateInstance(template, args, "my_vector")
        assert instance2.template_def is template
        assert len(instance2.arguments) == 3
        assert instance2.instance_name == "my_vector"
    
    def test_instance_string_representation(self):
        """Test the string representation of a template instance."""
        # Create a template definition
        params = [
            TemplateParameter("x", 1),
            TemplateParameter("y", 2),
            TemplateParameter("z", 3)
        ]
        template = TemplateDefinition("Vector3", params)
        
        # Create arguments
        args = [
            TemplateArgument(10),
            TemplateArgument(20, "y"),
            TemplateArgument(30, "z")
        ]
        
        # Create instance without name
        instance1 = TemplateInstance(template, args)
        assert str(instance1) == "Vector3(10, y=20, z=30)"
        
        # Create instance with name
        instance2 = TemplateInstance(template, args, "my_vector")
        assert str(instance2) == "my_vector: Vector3(10, y=20, z=30)"
    
    def test_get_argument_by_name(self):
        """Test getting an argument by parameter name."""
        # Create a template definition
        params = [
            TemplateParameter("x", 1),
            TemplateParameter("y", 2),
            TemplateParameter("z", 3)
        ]
        template = TemplateDefinition("Vector3", params)
        
        # Create arguments
        args = [
            TemplateArgument(10),
            TemplateArgument(20, "y"),
            TemplateArgument(30, "z")
        ]
        
        # Create instance
        instance = TemplateInstance(template, args)
        
        # Get named arguments
        assert instance.get_argument_by_name("y").value == 20
        assert instance.get_argument_by_name("z").value == 30
        
        # Get non-existent argument
        assert instance.get_argument_by_name("x") is None
        assert instance.get_argument_by_name("w") is None
    
    def test_get_argument_by_position(self):
        """Test getting an argument by position."""
        # Create a template definition
        params = [
            TemplateParameter("x", 1),
            TemplateParameter("y", 2),
            TemplateParameter("z", 3)
        ]
        template = TemplateDefinition("Vector3", params)
        
        # Create arguments
        args = [
            TemplateArgument(10),
            TemplateArgument(20),
            TemplateArgument(30, "z")
        ]
        
        # Create instance
        instance = TemplateInstance(template, args)
        
        # Get positional arguments
        assert instance.get_argument_by_position(0).value == 10
        assert instance.get_argument_by_position(1).value == 20
        
        # Get non-existent argument
        assert instance.get_argument_by_position(2) is None
        assert instance.get_argument_by_position(-1) is None


class TestTemplateRegistry:
    """Tests for the TemplateRegistry class."""
    
    def test_registry_creation(self):
        """Test creating a template registry."""
        registry = TemplateRegistry()
        assert len(registry.templates) == 0
    
    def test_register_get_template(self):
        """Test registering and getting a template."""
        registry = TemplateRegistry()
        
        # Create a template
        template = TemplateDefinition("Vector3", [
            TemplateParameter("x", 0),
            TemplateParameter("y", 0),
            TemplateParameter("z", 0)
        ])
        
        # Register the template
        registry.register(template)
        assert len(registry.templates) == 1
        assert "Vector3" in registry.templates
        
        # Get the template
        retrieved = registry.get_template("Vector3")
        assert retrieved is template
        
        # Get a non-existent template
        assert registry.get_template("Matrix3") is None
        
        # Try to register a template with the same name
        with pytest.raises(ValueError):
            registry.register(TemplateDefinition("Vector3"))
    
    def test_instantiate(self):
        """Test instantiating a template."""
        registry = TemplateRegistry()
        
        # Create a template
        template = TemplateDefinition("Vector3", [
            TemplateParameter("x", 0),
            TemplateParameter("y", 0),
            TemplateParameter("z", 0)
        ])
        
        # Register the template
        registry.register(template)
        
        # Create arguments
        args = [
            TemplateArgument(1),
            TemplateArgument(2, "y"),
            TemplateArgument(3, "z")
        ]
        
        # Instantiate the template
        instance = registry.instantiate("Vector3", args, "my_vector")
        assert instance.template_def is template
        assert len(instance.arguments) == 3
        assert instance.instance_name == "my_vector"
        
        # Try to instantiate a non-existent template
        with pytest.raises(ValueError):
            registry.instantiate("Matrix3", args)
    
    def test_validate_arguments(self):
        """Test validating template arguments."""
        registry = TemplateRegistry()
        
        # Create a template with required and optional parameters
        template = TemplateDefinition("Test", [
            TemplateParameter("a"),  # Required
            TemplateParameter("b"),  # Required
            TemplateParameter("c", 3)  # Optional
        ])
        
        # Register the template
        registry.register(template)
        
        # Valid arguments (all required parameters provided)
        args1 = [
            TemplateArgument(1),  # a
            TemplateArgument(2)   # b
        ]
        instance1 = registry.instantiate("Test", args1)
        assert instance1 is not None
        
        # Valid arguments (all parameters provided)
        args2 = [
            TemplateArgument(1),       # a
            TemplateArgument(2),       # b
            TemplateArgument(3, "c")   # c (named)
        ]
        instance2 = registry.instantiate("Test", args2)
        assert instance2 is not None
        
        # Valid arguments (required parameters provided with different ordering)
        args3 = [
            TemplateArgument(2, "b"),  # b (named)
            TemplateArgument(1, "a")   # a (named)
        ]
        instance3 = registry.instantiate("Test", args3)
        assert instance3 is not None
        
        # Invalid arguments (missing required parameter)
        args4 = [
            TemplateArgument(1)  # a
        ]
        with pytest.raises(ValueError):
            registry.instantiate("Test", args4)
        
        # Invalid arguments (unknown parameter)
        args5 = [
            TemplateArgument(1),       # a
            TemplateArgument(2),       # b
            TemplateArgument(4, "d")   # d (unknown)
        ]
        with pytest.raises(ValueError):
            registry.instantiate("Test", args5)
        
        # Invalid arguments (too many positional arguments)
        args6 = [
            TemplateArgument(1),  # a
            TemplateArgument(2),  # b
            TemplateArgument(3),  # c
            TemplateArgument(4)   # too many
        ]
        with pytest.raises(ValueError):
            registry.instantiate("Test", args6)


class TestTemplateParsing:
    """Tests for the template parsing functions."""
    
    def test_parse_template_declaration(self):
        """Test parsing a template declaration."""
        # Simple template with no parameters
        template1 = parse_template_declaration("template Empty()")
        assert template1.name == "Empty"
        assert len(template1.parameters) == 0
        
        # Template with parameters
        template2 = parse_template_declaration("template Vector3(x=0.0, y=0.0, z=0.0)")
        assert template2.name == "Vector3"
        assert len(template2.parameters) == 3
        assert template2.parameters[0].name == "x"
        assert template2.parameters[0].default_value == "0.0"
        assert template2.parameters[1].name == "y"
        assert template2.parameters[1].default_value == "0.0"
        assert template2.parameters[2].name == "z"
        assert template2.parameters[2].default_value == "0.0"
        
        # Template with typed parameters
        template3 = parse_template_declaration("template Point(x: float, y: float)")
        assert template3.name == "Point"
        assert len(template3.parameters) == 2
        assert template3.parameters[0].name == "x"
        assert template3.parameters[0].parameter_type == "float"
        assert template3.parameters[0].default_value is None
        assert template3.parameters[1].name == "y"
        assert template3.parameters[1].parameter_type == "float"
        assert template3.parameters[1].default_value is None
        
        # Template with mixed parameters
        template4 = parse_template_declaration("template Mixed(a, b: int, c=3, d: float=4.0)")
        assert template4.name == "Mixed"
        assert len(template4.parameters) == 4
        assert template4.parameters[0].name == "a"
        assert template4.parameters[0].parameter_type is None
        assert template4.parameters[0].default_value is None
        assert template4.parameters[1].name == "b"
        assert template4.parameters[1].parameter_type == "int"
        assert template4.parameters[1].default_value is None
        assert template4.parameters[2].name == "c"
        assert template4.parameters[2].parameter_type is None
        assert template4.parameters[2].default_value == "3"
        assert template4.parameters[3].name == "d"
        assert template4.parameters[3].parameter_type == "float"
        assert template4.parameters[3].default_value == "4.0"
        
        # Invalid template declaration
        with pytest.raises(ValueError):
            parse_template_declaration("not a template")
    
    def test_parse_template_instantiation(self):
        """Test parsing a template instantiation."""
        # Simple instantiation with no arguments
        name1, args1 = parse_template_instantiation("Empty()")
        assert name1 == "Empty"
        assert len(args1) == 0
        
        # Instantiation with positional arguments
        name2, args2 = parse_template_instantiation("Vector3(1, 2, 3)")
        assert name2 == "Vector3"
        assert len(args2) == 3
        assert args2[0].value == "1"
        assert args2[0].name is None
        assert args2[1].value == "2"
        assert args2[1].name is None
        assert args2[2].value == "3"
        assert args2[2].name is None
        
        # Instantiation with named arguments
        name3, args3 = parse_template_instantiation("Point(x=10, y=20)")
        assert name3 == "Point"
        assert len(args3) == 2
        assert args3[0].value == "10"
        assert args3[0].name == "x"
        assert args3[1].value == "20"
        assert args3[1].name == "y"
        
        # Instantiation with mixed arguments
        name4, args4 = parse_template_instantiation("Mixed(1, b=2, c=3)")
        assert name4 == "Mixed"
        assert len(args4) == 3
        assert args4[0].value == "1"
        assert args4[0].name is None
        assert args4[1].value == "2"
        assert args4[1].name == "b"
        assert args4[2].value == "3"
        assert args4[2].name == "c"
        
        # Invalid template instantiation
        with pytest.raises(ValueError):
            parse_template_instantiation("not an instantiation")
