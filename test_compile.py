import py_compile
import sys

try:
    py_compile.compile('ingest_pdf/pipeline/ingest_text_like.py', doraise=True)
    print("✓ ingest_text_like.py compiles successfully!")
except py_compile.PyCompileError as e:
    print(f"✗ Compilation error: {e}")
    sys.exit(1)
