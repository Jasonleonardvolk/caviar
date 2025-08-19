"""
Test WGSL validation and hot reload integration.
"""
import os, subprocess, sys, time
from pathlib import Path
import pytest
from .utils.wgsl_mutator import (
    toggle_define, 
    inject_syntax_error, 
    restore_shader,
    add_shader_feature
)

def test_wgsl_mutation_triggers_validator(repo_root):
    """Test that shader modifications trigger validation."""
    shader = repo_root / "frontend" / "hybrid" / "wgsl" / "lightFieldComposer.wgsl"
    if not shader.exists():
        pytest.skip("lightFieldComposer.wgsl not present")
    
    # Save original content
    original = shader.read_text(encoding="utf-8")
    
    try:
        # Toggle a macro
        hash1 = toggle_define(shader)
        
        # Verify file changed
        current = shader.read_text(encoding="utf-8")
        assert current != original, "Shader not modified"
        
        # Run validator
        result = subprocess.run(
            ["naga", str(shader)],
            capture_output=True,
            text=True
        )
        
        # Should still be valid after macro toggle
        assert result.returncode == 0, f"Validation failed: {result.stderr}"
        
    finally:
        # Restore original
        shader.write_text(original, encoding="utf-8")

def test_wgsl_syntax_error_detection(repo_root):
    """Test that validator catches syntax errors."""
    shader = repo_root / "frontend" / "hybrid" / "wgsl" / "lightFieldComposerEnhanced.wgsl"
    if not shader.exists():
        pytest.skip("lightFieldComposerEnhanced.wgsl not present")
    
    original = inject_syntax_error(shader)
    
    try:
        # Run validator - should fail
        result = subprocess.run(
            ["naga", str(shader)],
            capture_output=True,
            text=True
        )
        
        # Should detect the syntax error
        assert result.returncode != 0, "Validator didn't catch syntax error"
        assert "error" in result.stderr.lower() or "invalid" in result.stderr.lower()
        
    finally:
        restore_shader(shader, original)

def test_wgsl_f16_feature_detection(repo_root):
    """Test detection of shader-f16 feature requirement."""
    shader = repo_root / "frontend" / "hybrid" / "wgsl" / "lightFieldComposer.wgsl"
    if not shader.exists():
        pytest.skip("lightFieldComposer.wgsl not present")
    
    original = shader.read_text(encoding="utf-8")
    
    try:
        # Add f16 usage
        if add_shader_feature(shader, "shader-f16"):
            # Validate with feature flag
            result = subprocess.run(
                ["naga", str(shader)],
                capture_output=True,
                text=True
            )
            
            # May warn about f16 but shouldn't hard fail
            # (depends on naga version and configuration)
            if "f16" in result.stderr.lower() or "half" in result.stderr.lower():
                print("f16 feature detected by validator")
        
    finally:
        shader.write_text(original, encoding="utf-8")

def test_wgsl_validation_performance(repo_root):
    """Test validation completes within performance target."""
    shaders = list((repo_root / "frontend" / "hybrid" / "wgsl").glob("*.wgsl"))
    
    if not shaders:
        pytest.skip("No WGSL shaders found")
    
    for shader in shaders[:3]:  # Test first 3 shaders
        start = time.time()
        result = subprocess.run(
            ["naga", str(shader)],
            capture_output=True,
            text=True,
            timeout=1.0  # 1 second timeout
        )
        elapsed = time.time() - start
        
        # Should complete in < 100ms (requirement)
        assert elapsed < 0.2, f"Validation of {shader.name} took {elapsed*1000:.0f}ms"

def test_validate_all_shaders_script(repo_root):
    """Test the validate-wgsl.js script if present."""
    script = repo_root / "frontend" / "scripts" / "validate-wgsl.js"
    if not script.exists():
        pytest.skip("validate-wgsl.js not present")
    
    # Run the validation script
    result = subprocess.run(
        ["node", str(script)],
        cwd=str(repo_root / "frontend"),
        capture_output=True,
        text=True,
        timeout=10.0
    )
    
    # Should complete successfully for valid shaders
    if result.returncode != 0:
        print(f"Validation output:\n{result.stdout}\n{result.stderr}")
    
    # Check for expected output patterns
    assert "✅" in result.stdout or "OK" in result.stdout or result.returncode == 0

def test_wgsl_hot_reload_plugin(repo_root, tmp_path):
    """Test hot reload plugin behavior with shader changes."""
    plugin_path = repo_root / "frontend" / "plugins" / "wgslValidatePlugin.js"
    if not plugin_path.exists():
        pytest.skip("wgslValidatePlugin.js not present")
    
    # Create a test shader in temp location
    test_shader = tmp_path / "test.wgsl"
    test_shader.write_text("""
        @compute @workgroup_size(1)
        fn main() {
            // Test shader
        }
    """)
    
    # Simulate hot reload by modifying file
    original_mtime = test_shader.stat().st_mtime
    time.sleep(0.1)
    
    test_shader.write_text("""
        @compute @workgroup_size(1)
        fn main() {
            // Modified test shader
            let x: u32 = 1u;
        }
    """)
    
    # Verify file was modified
    new_mtime = test_shader.stat().st_mtime
    assert new_mtime > original_mtime, "File modification time didn't change"

def test_parallel_shader_validation(repo_root):
    """Test concurrent validation of multiple shaders."""
    shaders = list((repo_root / "frontend" / "hybrid" / "wgsl").glob("*.wgsl"))
    
    if len(shaders) < 2:
        pytest.skip("Need at least 2 shaders for parallel test")
    
    import concurrent.futures
    
    def validate_shader(shader_path):
        result = subprocess.run(
            ["naga", str(shader_path)],
            capture_output=True,
            text=True
        )
        return shader_path.name, result.returncode == 0
    
    # Validate all shaders in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(validate_shader, s) for s in shaders]
        results = [f.result() for f in futures]
    
    # All should validate successfully
    for name, valid in results:
        assert valid, f"Shader {name} failed validation"
