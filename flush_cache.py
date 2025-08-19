import compileall
import pathlib

# Recompile all Python files in python/core
compileall.compile_dir("str(PROJECT_ROOT / "python\\core", force=True, quiet=1)

# Remove all .pyc files
for p in pathlib.Path("str(PROJECT_ROOT / "python\\core").rglob("*.pyc"):
    p.unlink(missing_ok=True)

print("Bytecode cache flushed successfully!")
