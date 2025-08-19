#!/usr/bin/env python3
"""
ROBUST TORI SYSTEM FIX
Complete solution to fix all startup issues and build a stable foundation
"""

import os
import sys
import json
import shutil
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
import ast
import re

class TORISystemFixer:
    """Comprehensive TORI system fixer"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.fixes_applied = []
        self.errors_found = []
        self.backup_dir = self.root_dir / f"backups_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backup_dir.mkdir(exist_ok=True)
        
    def backup_file(self, filepath):
        """Create backup before modifying"""
        if os.path.exists(filepath):
            rel_path = Path(filepath).relative_to(self.root_dir)
            backup_path = self.backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filepath, backup_path)
            return backup_path
        return None

    def create_complete_concept_mesh_stub(self):
        """Create a complete concept_mesh_rs stub module"""
        print("\n🔧 Creating complete concept_mesh_rs stub module...")
        
        # Create the package structure
        concept_mesh_dir = self.root_dir / "concept_mesh_rs"
        concept_mesh_dir.mkdir(exist_ok=True)
        
        # __init__.py with all exports
        init_content = '''"""
Concept Mesh Rust Stub
Complete implementation of expected interface
"""

from .interface import ConceptMesh, ConceptMeshError
from .loader import ConceptMeshLoader
from .types import MemoryEntry, MemoryQuery, PhaseTag, Concept, ConceptRelation

__all__ = [
    'ConceptMesh',
    'ConceptMeshError',
    'ConceptMeshLoader',
    'MemoryEntry',
    'MemoryQuery',
    'PhaseTag',
    'Concept',
    'ConceptRelation'
]

# Version info
__version__ = "0.1.0"
RUST_WHEEL = False  # This is a Python stub
'''
        
        with open(concept_mesh_dir / "__init__.py", 'w', encoding='utf-8') as f:
            f.write(init_content)
        
        # interface.py
        interface_content = '''"""
ConceptMesh Interface
Main interface for concept mesh operations
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class ConceptMeshError(Exception):
    """Base exception for concept mesh errors"""
    pass

class ConceptMesh:
    """
    Main ConceptMesh interface
    Accepts no parameters in constructor to match Rust interface
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to avoid multiple instances"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize without parameters"""
        if not hasattr(self, 'initialized'):
            self.concepts = {}
            self.relations = []
            self.storage_path = Path("data/concept_mesh")
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.initialized = True
            self._load_data()
    
    def _load_data(self):
        """Load existing data from disk"""
        data_file = self.storage_path / "concepts.json"
        if data_file.exists():
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.concepts = data.get('concepts', {})
                        self.relations = data.get('relations', [])
            except Exception as e:
                print(f"Warning: Failed to load concept data: {e}")
    
    def _save_data(self):
        """Save data to disk"""
        data_file = self.storage_path / "concepts.json"
        try:
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'concepts': self.concepts,
                    'relations': self.relations,
                    'version': '1.0'
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save concept data: {e}")
    
    def add_concept(self, concept_id: str, name: str = None, **kwargs):
        """Add a concept to the mesh"""
        self.concepts[concept_id] = {
            'id': concept_id,
            'name': name or concept_id,
            'created_at': datetime.now().isoformat(),
            **kwargs
        }
        self._save_data()
        return concept_id
    
    def get_concept(self, concept_id: str):
        """Get a concept by ID"""
        return self.concepts.get(concept_id)
    
    def search_concepts(self, query: str, limit: int = 10):
        """Search concepts by name or ID"""
        results = []
        query_lower = query.lower()
        
        for concept_id, concept in self.concepts.items():
            if (query_lower in concept.get('name', '').lower() or 
                query_lower in concept_id.lower()):
                results.append(concept)
                if len(results) >= limit:
                    break
        
        return results
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str):
        """Add a relation between concepts"""
        relation = {
            'source': source_id,
            'target': target_id,
            'type': relation_type,
            'created_at': datetime.now().isoformat()
        }
        self.relations.append(relation)
        self._save_data()
        return relation
    
    def get_relations(self, concept_id: str):
        """Get all relations for a concept"""
        return [r for r in self.relations 
                if r['source'] == concept_id or r['target'] == concept_id]
    
    def clear(self):
        """Clear all concepts and relations"""
        self.concepts = {}
        self.relations = []
        self._save_data()
    
    def __repr__(self):
        return f"ConceptMesh(concepts={len(self.concepts)}, relations={len(self.relations)})"
'''
        
        with open(concept_mesh_dir / "interface.py", 'w', encoding='utf-8') as f:
            f.write(interface_content)
        
        # loader.py
        loader_content = '''"""
ConceptMesh Loader
Handles loading concept mesh data from various sources
"""

from .interface import ConceptMesh
import json
from pathlib import Path

class ConceptMeshLoader:
    """Loader for concept mesh data"""
    
    def __init__(self):
        self.loaded = False
        self.mesh = None
        self.source_path = None
    
    def load(self, path=None):
        """Load concept mesh from path"""
        self.source_path = path
        self.mesh = ConceptMesh()
        
        if path and Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load concepts
                if 'concepts' in data:
                    for concept_id, concept_data in data['concepts'].items():
                        self.mesh.add_concept(concept_id, **concept_data)
                
                # Load relations
                if 'relations' in data:
                    for relation in data['relations']:
                        self.mesh.add_relation(
                            relation['source'],
                            relation['target'],
                            relation.get('type', 'related')
                        )
                
                self.loaded = True
            except Exception as e:
                print(f"Warning: Failed to load from {path}: {e}")
        
        return self
    
    def get_mesh(self):
        """Get the concept mesh instance"""
        if not self.mesh:
            self.mesh = ConceptMesh()
        return self.mesh
    
    def save(self, path=None):
        """Save the mesh to a file"""
        save_path = path or self.source_path
        if save_path and self.mesh:
            data = {
                'concepts': self.mesh.concepts,
                'relations': self.mesh.relations,
                'version': '1.0'
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
'''
        
        with open(concept_mesh_dir / "loader.py", 'w', encoding='utf-8') as f:
            f.write(loader_content)
        
        # types.py
        types_content = '''"""
ConceptMesh Types
Data structures used by the concept mesh
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

@dataclass
class MemoryEntry:
    """A memory entry in the concept mesh"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    strength: float = 0.7
    timestamp: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        if isinstance(self.timestamp, datetime):
            data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def to_json(self):
        """Convert to JSON"""
        return json.dumps(self.to_dict())

@dataclass
class MemoryQuery:
    """Query for searching memories"""
    text: str
    limit: int = 5
    min_strength: float = 0.3
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)

@dataclass
class PhaseTag:
    """Phase information for quantum-inspired operations"""
    phase: float
    amplitude: float = 0.5
    frequency: float = 0.5
    coherence: float = 1.0
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Concept:
    """A concept in the mesh"""
    id: str
    name: str
    description: str = ""
    category: str = "general"
    importance: float = 1.0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        data = asdict(self)
        if isinstance(self.created_at, datetime):
            data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class ConceptRelation:
    """Relationship between concepts"""
    source_id: str
    target_id: str
    relation_type: str
    strength: float = 1.0
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        data = asdict(self)
        if isinstance(self.created_at, datetime):
            data['created_at'] = self.created_at.isoformat()
        return data
'''
        
        with open(concept_mesh_dir / "types.py", 'w', encoding='utf-8') as f:
            f.write(types_content)
        
        self.fixes_applied.append("Created complete concept_mesh_rs stub module")
        print("✅ Complete concept_mesh_rs stub created")

    def fix_import_statements(self):
        """Fix import statements across the codebase"""
        print("\n🔧 Fixing import statements...")
        
        # Files known to have import issues
        files_to_fix = [
            "mcp_metacognitive/core/soliton_memory.py",
            "ingest_pdf/pipeline/quality.py",
            "python/core/__init__.py"
        ]
        
        for file_path in files_to_fix:
            if not os.path.exists(file_path):
                continue
            
            print(f"  Fixing {file_path}...")
            self.backup_file(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix the imports to use the stub
            original = content
            
            # Remove attempts to import from non-existent modules
            content = re.sub(
                r'from concept_mesh_rs\.interface import ConceptMesh\s*\n\s*from concept_mesh_rs\.types import.*\n',
                'from concept_mesh_rs import ConceptMesh, MemoryEntry, MemoryQuery, PhaseTag\n',
                content
            )
            
            # Fix CONCEPT_MESH_AVAILABLE checks
            content = re.sub(
                r'CONCEPT_MESH_AVAILABLE = True\s*\n\s*USING_RUST_WHEEL = True',
                'CONCEPT_MESH_AVAILABLE = True\nUSING_RUST_WHEEL = False  # Using stub',
                content
            )
            
            if content != original:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixes_applied.append(f"Fixed imports in {file_path}")

    def create_missing_directories(self):
        """Create missing directories with proper __init__.py files"""
        print("\n🔧 Creating missing directories...")
        
        directories = [
            "mcp_metacognitive",
            "mcp_metacognitive/core",
            "mcp_metacognitive/resources",
            "mcp_metacognitive/tools",
            "ingest_pdf",
            "ingest_pdf/pipeline",
            "python",
            "python/core",
            "python/hardware",
            "python/stability",
            "cog",
            "data",
            "data/concept_mesh"
        ]
        
        for dir_path in directories:
            dir_full = self.root_dir / dir_path
            dir_full.mkdir(parents=True, exist_ok=True)
            
            init_file = dir_full / "__init__.py"
            if not init_file.exists():
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(f'"""Package: {dir_path}"""\n')
                self.fixes_applied.append(f"Created {init_file}")

    def fix_environment_variables(self):
        """Set required environment variables"""
        print("\n🔧 Setting environment variables...")
        
        env_vars = {
            'TORI_DISABLE_ENTROPY_PRUNE': '1',
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONUTF8': '1',
            'TORI_CONCEPT_MESH_STUB': '1'
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            self.fixes_applied.append(f"Set {var}={value}")

    def install_missing_dependencies(self):
        """Install missing Python dependencies"""
        print("\n🔧 Checking Python dependencies...")
        
        required_packages = [
            'numpy',
            'networkx',
            'dataclasses',
            'typing-extensions',
            'python-dateutil'
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                print(f"  Installing {package}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             capture_output=True, text=True)
                self.fixes_applied.append(f"Installed {package}")

    def create_diagnostic_report(self):
        """Create a comprehensive diagnostic report"""
        report_content = f"""# TORI System Fix Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Fixes Applied

Total fixes: {len(self.fixes_applied)}

### Details:
"""
        
        for fix in self.fixes_applied:
            report_content += f"- {fix}\n"
        
        if self.errors_found:
            report_content += f"\n## Errors Found\n\n"
            for error in self.errors_found:
                report_content += f"- {error}\n"
        
        report_content += """
## Next Steps

1. Run `poetry run python enhanced_launcher.py`
2. The system should start with minimal warnings
3. Check the `TORI_SYSTEM_STATUS.json` file for runtime status

## Verification Commands

```bash
# Test imports
python -c "from concept_mesh_rs import ConceptMesh; print(ConceptMesh())"

# Launch TORI
poetry run python enhanced_launcher.py
```

## Backup Location

All modified files have been backed up to: `{}`
""".format(self.backup_dir)
        
        report_path = self.root_dir / "TORI_FIX_REPORT.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📄 Report saved to: {report_path}")

    def create_system_status_monitor(self):
        """Create a system status monitor"""
        print("\n🔧 Creating system status monitor...")
        
        monitor_content = '''#!/usr/bin/env python3
"""
TORI System Status Monitor
Tracks the health of the TORI system
"""

import json
import psutil
import subprocess
from datetime import datetime
from pathlib import Path

def check_system_status():
    """Check various system components"""
    status = {
        "timestamp": datetime.now().isoformat(),
        "components": {},
        "resources": {},
        "errors": []
    }
    
    # Check Python imports
    imports_ok = True
    try:
        from concept_mesh_rs import ConceptMesh
        cm = ConceptMesh()
        status["components"]["concept_mesh"] = "OK"
    except Exception as e:
        status["components"]["concept_mesh"] = f"ERROR: {str(e)}"
        imports_ok = False
    
    # Check system resources
    status["resources"]["cpu_percent"] = psutil.cpu_percent()
    status["resources"]["memory_percent"] = psutil.virtual_memory().percent
    status["resources"]["disk_free_gb"] = psutil.disk_usage('/').free / (1024**3)
    
    # Check if enhanced launcher works
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import enhanced_launcher"],
            capture_output=True,
            text=True,
            timeout=5
        )
        status["components"]["enhanced_launcher"] = "OK" if result.returncode == 0 else "ERROR"
    except Exception as e:
        status["components"]["enhanced_launcher"] = f"ERROR: {str(e)}"
    
    # Overall status
    status["overall"] = "OK" if imports_ok else "DEGRADED"
    
    # Save status
    with open("TORI_SYSTEM_STATUS.json", "w") as f:
        json.dump(status, f, indent=2)
    
    return status

if __name__ == "__main__":
    import sys
    status = check_system_status()
    print(f"System Status: {status['overall']}")
    
    if status['overall'] != "OK":
        print("\\nIssues found:")
        for component, state in status['components'].items():
            if "ERROR" in state:
                print(f"  - {component}: {state}")
'''
        
        monitor_path = self.root_dir / "tori_status_monitor.py"
        with open(monitor_path, 'w', encoding='utf-8') as f:
            f.write(monitor_content)
        
        self.fixes_applied.append("Created system status monitor")

    def run_all_fixes(self):
        """Run all fixes in sequence"""
        print("🚀 ROBUST TORI SYSTEM FIX")
        print("=" * 60)
        
        try:
            # Core fixes
            self.create_complete_concept_mesh_stub()
            self.fix_import_statements()
            self.create_missing_directories()
            self.fix_environment_variables()
            self.install_missing_dependencies()
            self.create_system_status_monitor()
            
            # Generate report
            self.create_diagnostic_report()
            
            print("\n✅ All fixes applied successfully!")
            print(f"📊 Total fixes: {len(self.fixes_applied)}")
            
            # Run status check
            print("\n🔍 Running system status check...")
            subprocess.run([sys.executable, "tori_status_monitor.py"])
            
        except Exception as e:
            print(f"\n❌ Error during fix process: {e}")
            traceback.print_exc()
            self.errors_found.append(str(e))

def main():
    """Main execution"""
    fixer = TORISystemFixer()
    fixer.run_all_fixes()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Run: poetry run python enhanced_launcher.py")
    print("2. Check: TORI_FIX_REPORT.md for details")
    print("3. Monitor: TORI_SYSTEM_STATUS.json for runtime status")
    print("=" * 60)

if __name__ == "__main__":
    main()
