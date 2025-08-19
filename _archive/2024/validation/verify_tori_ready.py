"""
Quick diagnostic to verify TORI is ready to run
"""
import os
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import subprocess
import sys

print("🔍 TORI System Diagnostic")
print("=" * 50)

# Check concept files
print("\n📁 Concept Database Files:")
concept_files = [
    (Path(r"{PROJECT_ROOT}\ingest_pdf\data\concept_file_storage.json"), "Pipeline concepts"),
    (Path(r"{PROJECT_ROOT}\ingest_pdf\data\concept_seed_universal.json"), "Universal seeds"),
    (Path(r"{PROJECT_ROOT}\concept_mesh\concept_mesh_data.json"), "Mesh data"),
    (Path(r"{PROJECT_ROOT}\concept_mesh\concepts.json"), "Mesh concepts"),
]

total_concepts = 0
for file_path, desc in concept_files:
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    count = len(data.get('concepts', []))
                else:
                    count = 0
                total_concepts += count
                print(f"✅ {desc}: {file_path.name} ({count} items)")
        except:
            print(f"⚠️  {desc}: {file_path.name} (invalid JSON)")
    else:
        print(f"❌ {desc}: {file_path.name} (missing)")

print(f"\n📊 Total concepts available: {total_concepts}")

# Check Python dependencies
print("\n📦 Python Dependencies:")
deps = [
    ("cv2", "OpenCV", "opencv-python-headless"),
    ("pydub", "PyDub", "pydub"),
    ("pymupdf", "PyMuPDF", "pymupdf"),
]

missing_deps = []
for module, name, package in deps:
    try:
        __import__(module)
        print(f"✅ {name} installed")
    except ImportError:
        print(f"❌ {name} missing (install with: pip install {package})")
        missing_deps.append(package)

# Check ffmpeg
print("\n🎬 External Dependencies:")
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
if os.path.exists(ffmpeg_path):
    print("✅ ffmpeg installed")
else:
    print("❌ ffmpeg missing (required for audio processing)")

# Check environment
print("\n🌍 Environment:")
env_vars = ["CONCEPT_DB_PATH", "TORI_CONCEPT_DB", "CONCEPT_MESH_DIR"]
for var in env_vars:
    value = os.environ.get(var)
    if value:
        print(f"✅ {var} = {value}")
    else:
        print(f"⚠️  {var} not set")

# Summary
print("\n" + "=" * 50)
if missing_deps:
    print("⚠️  Missing dependencies. Install with:")
    print(f"   pip install {' '.join(missing_deps)}")
elif total_concepts == 0:
    print("⚠️  No concepts loaded. TORI will use fallback processing.")
    print("   If you have the original concept-mesh folder, copy it to:")
    print("   str(PROJECT_ROOT / "concept_mesh\\")
else:
    print("✅ TORI is ready to launch!")
    print("   Run: python enhanced_launcher.py")
