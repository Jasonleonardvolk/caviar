"""
End-to-end tests for the ELFIN module system.

This module contains more complex tests that exercise the entire module system,
including circular dependency detection and complex template instantiation.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path

from alan_backend.elfin.modules.resolver import ImportResolver, ModuleSearchPath
from alan_backend.elfin.modules.symbols import SymbolTable
from alan_backend.elfin.modules.templates import TemplateRegistry
from alan_backend.elfin.parser.module_parser import parse_elfin_module
from alan_backend.elfin.modules.errors import CircularDependencyError


class TestModuleSystemE2E:
    """End-to-end tests for the ELFIN module system."""
    
    def setup_method(self):
        """Set up a temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.resolver = ImportResolver([self.temp_dir])
        self.symbol_table = SymbolTable()
        self.template_registry = TemplateRegistry()
    
    def teardown_method(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def write_file(self, name, content):
        """Write content to a file in the temporary directory."""
        path = os.path.join(self.temp_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return path
    
    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected."""
        # Create a circular dependency between three files
        self.write_file("a.elfin", """
        import B from "b.elfin";
        
        concept A {
            // Uses B
        }
        """)
        
        self.write_file("b.elfin", """
        import C from "c.elfin";
        
        concept B {
            // Uses C
        }
        """)
        
        self.write_file("c.elfin", """
        import A from "a.elfin";
        
        concept C {
            // Uses A, creating a circular dependency
        }
        """)
        
        # Parse a.elfin, which should eventually try to import itself via c.elfin
        with open(os.path.join(self.temp_dir, "a.elfin"), "r") as f:
            source = f.read()
        
        # This should raise a CircularDependencyError
        with pytest.raises(CircularDependencyError):
            module = parse_elfin_module(
                source,
                file_path=os.path.join(self.temp_dir, "a.elfin"),
                resolver=self.resolver,
                symbol_table=self.symbol_table,
                template_registry=self.template_registry
            )
    
    def test_complex_template_instantiation(self):
        """Test complex template instantiation with nested templates."""
        # Create template files
        self.write_file("vector.elfin", """
        template Vector3(x=0.0, y=0.0, z=0.0) {
            parameters {
                x: float = x;
                y: float = y;
                z: float = z;
            }
            
            property magnitude {
                return (x^2 + y^2 + z^2)^0.5;
            }
            
            property normalized {
                mag = magnitude;
                if (mag > 0) {
                    return Vector3(x/mag, y/mag, z/mag);
                } else {
                    return Vector3(0, 0, 0);
                }
            }
        }
        """)
        
        self.write_file("matrix.elfin", """
        import Vector3 from "vector.elfin";
        
        template Matrix3(
            row1=Vector3(1, 0, 0),
            row2=Vector3(0, 1, 0),
            row3=Vector3(0, 0, 1)
        ) {
            parameters {
                row1: Vector3 = row1;
                row2: Vector3 = row2;
                row3: Vector3 = row3;
            }
            
            property determinant {
                // Compute the determinant of the 3x3 matrix
                a = row1.x;
                b = row1.y;
                c = row1.z;
                d = row2.x;
                e = row2.y;
                f = row2.z;
                g = row3.x;
                h = row3.y;
                i = row3.z;
                
                return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
            }
            
            property transpose {
                return Matrix3(
                    Vector3(row1.x, row2.x, row3.x),
                    Vector3(row1.y, row2.y, row3.y),
                    Vector3(row1.z, row2.z, row3.z)
                );
            }
        }
        """)
        
        self.write_file("transform.elfin", """
        import Vector3 from "vector.elfin";
        import Matrix3 from "matrix.elfin";
        
        template Transform(
            rotation=Matrix3(),
            translation=Vector3()
        ) {
            parameters {
                rotation: Matrix3 = rotation;
                translation: Vector3 = translation;
            }
            
            function apply(point) {
                rotated = rotation.apply(point);
                return Vector3(
                    rotated.x + translation.x,
                    rotated.y + translation.y,
                    rotated.z + translation.z
                );
            }
            
            property inverse {
                inv_rotation = rotation.inverse;
                inv_translation = inv_rotation.apply(Vector3(
                    -translation.x,
                    -translation.y,
                    -translation.z
                ));
                
                return Transform(inv_rotation, inv_translation);
            }
        }
        """)
        
        self.write_file("main.elfin", """
        import Vector3 from "vector.elfin";
        import Matrix3 from "matrix.elfin";
        import Transform from "transform.elfin";
        
        concept Robot {
            property position = Vector3(0, 0, 0);
            property orientation = Matrix3();
            property pose = Transform(orientation, position);
            
            function move(dx, dy, dz) {
                position = Vector3(position.x + dx, position.y + dy, position.z + dz);
                pose = Transform(orientation, position);
            }
            
            function rotate(matrix) {
                orientation = matrix;
                pose = Transform(orientation, position);
            }
        }
        """)
        
        # Parse the main file
        with open(os.path.join(self.temp_dir, "main.elfin"), "r") as f:
            source = f.read()
        
        # This should successfully parse with complex template instantiations
        module = parse_elfin_module(
            source,
            file_path=os.path.join(self.temp_dir, "main.elfin"),
            resolver=self.resolver,
            symbol_table=self.symbol_table,
            template_registry=self.template_registry
        )
        
        # Check that all templates were registered
        assert self.template_registry.get_template("Vector3") is not None
        assert self.template_registry.get_template("Matrix3") is not None
        assert self.template_registry.get_template("Transform") is not None
        
        # Check that all symbols were imported
        assert self.symbol_table.lookup("Vector3") is not None
        assert self.symbol_table.lookup("Matrix3") is not None
        assert self.symbol_table.lookup("Transform") is not None
    
    def test_template_validation(self):
        """Test template validation with incorrect arguments."""
        # Create a template file
        self.write_file("controller.elfin", """
        template PIDController(kp, ki=0.0, kd=0.0) {
            parameters {
                kp: float;  // Required parameter
                ki: float = ki;
                kd: float = kd;
            }
            
            function compute(error, dt) {
                // PID computation
                return kp*error + ki*integral + kd*derivative;
            }
        }
        """)
        
        # First create a file with correct instantiation
        self.write_file("correct.elfin", """
        import PIDController from "controller.elfin";
        
        concept Robot {
            property controller = PIDController(1.0);  // Correct
        }
        """)
        
        # Then create a file with incorrect instantiation (missing required parameter)
        self.write_file("incorrect.elfin", """
        import PIDController from "controller.elfin";
        
        concept Robot {
            property controller = PIDController();  // Missing required kp
        }
        """)
        
        # Parse the correct file - should succeed
        with open(os.path.join(self.temp_dir, "correct.elfin"), "r") as f:
            correct_source = f.read()
        
        correct_module = parse_elfin_module(
            correct_source,
            file_path=os.path.join(self.temp_dir, "correct.elfin"),
            resolver=self.resolver,
            symbol_table=self.symbol_table,
            template_registry=self.template_registry
        )
        
        # Parse the incorrect file - should fail with a parse error
        with open(os.path.join(self.temp_dir, "incorrect.elfin"), "r") as f:
            incorrect_source = f.read()
        
        with pytest.raises(Exception):  # Should raise some kind of parse error
            incorrect_module = parse_elfin_module(
                incorrect_source,
                file_path=os.path.join(self.temp_dir, "incorrect.elfin"),
                resolver=self.resolver,
                symbol_table=SymbolTable(),  # Use a new symbol table
                template_registry=self.template_registry
            )
