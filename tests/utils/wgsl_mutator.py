"""
WGSL file mutation utilities for testing shader validation.
"""
from pathlib import Path
import time
import hashlib

def toggle_define(shader_path: Path, macro: str = "/*TOGGLE*/"):
    """Toggle a macro in shader file to trigger validation."""
    text = shader_path.read_text(encoding="utf-8")
    if macro in text:
        text = text.replace(macro, "")
    else:
        text = text + f"\n{macro}\n"
    shader_path.write_text(text, encoding="utf-8")
    time.sleep(0.05)  # give watcher a tick
    return hashlib.md5(text.encode()).hexdigest()

def inject_syntax_error(shader_path: Path) -> str:
    """Inject a syntax error to test validation catches it."""
    text = shader_path.read_text(encoding="utf-8")
    original = text
    
    # Find a struct and break it
    if "struct" in text and "," in text:
        # Replace a comma with semicolon to break WGSL syntax
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "," in line and "//" not in line:  # Avoid comments
                lines[i] = line.replace(",", ";", 1)
                break
        text = "\n".join(lines)
    else:
        # Just add invalid syntax
        text += "\nINVALID SYNTAX ERROR TEST;\n"
    
    shader_path.write_text(text, encoding="utf-8")
    return original

def restore_shader(shader_path: Path, original_content: str):
    """Restore shader to original content."""
    shader_path.write_text(original_content, encoding="utf-8")

def add_shader_feature(shader_path: Path, feature: str = "shader-f16") -> bool:
    """Add a shader feature requirement to test capability detection."""
    text = shader_path.read_text(encoding="utf-8")
    
    if feature == "shader-f16":
        # Add f16 type usage
        if "f16" not in text:
            text = f"// Testing f16 support\nvar<private> test_f16: f16 = 0.0h;\n" + text
            shader_path.write_text(text, encoding="utf-8")
            return True
    
    return False
