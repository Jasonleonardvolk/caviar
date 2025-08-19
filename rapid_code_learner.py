from pigpen_config import PROJECT_ROOT
#!/usr/bin/env python3
"""
RAPID CODE LEARNING FOR TONKA
Fast track to badass coding without tokens/transformers
Using concept mesh + LSTM pattern learning
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re

class RapidCodeLearner:
    """
    Fast code learning system for TONKA
    Extracts patterns and stores in concept mesh (no tokens!)
    """
    
    def __init__(self, concept_mesh_dir: Path):
        self.concept_mesh_dir = concept_mesh_dir
        self.code_patterns_dir = concept_mesh_dir / "code_patterns"
        self.code_patterns_dir.mkdir(exist_ok=True)
        
        # Pattern categories optimized for LSTM learning
        self.pattern_types = {
            "function_patterns": [],
            "class_patterns": [],
            "error_patterns": [],
            "async_patterns": [],
            "api_patterns": [],
            "test_patterns": [],
            "architecture_patterns": []
        }
    
    def extract_patterns_from_file(self, file_path: Path) -> Dict[str, List[Dict]]:
        """Extract code patterns from a Python file"""
        patterns = {k: [] for k in self.pattern_types.keys()}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract different pattern types
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    pattern = self.extract_function_pattern(node, content)
                    patterns["function_patterns"].append(pattern)
                    
                    if node.name.startswith("test_"):
                        patterns["test_patterns"].append(pattern)
                    
                    if any(d.decorator_list for d in [node] if hasattr(d, 'decorator_list')):
                        if self.is_async_function(node):
                            patterns["async_patterns"].append(pattern)
                
                elif isinstance(node, ast.ClassDef):
                    pattern = self.extract_class_pattern(node, content)
                    patterns["class_patterns"].append(pattern)
                    
                    if "Server" in node.name or "API" in node.name:
                        patterns["api_patterns"].append(pattern)
                
                elif isinstance(node, ast.Try):
                    pattern = self.extract_error_pattern(node, content)
                    patterns["error_patterns"].append(pattern)
            
            # Extract architecture patterns from imports and structure
            arch_patterns = self.extract_architecture_patterns(content)
            patterns["architecture_patterns"].extend(arch_patterns)
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return patterns
    
    def extract_function_pattern(self, node: ast.FunctionDef, source: str) -> Dict:
        """Extract function pattern for concept mesh"""
        # Get function source
        func_source = ast.get_source_segment(source, node) or ""
        
        # Extract key components
        pattern = {
            "name": node.name,
            "type": "function",
            "async": self.is_async_function(node),
            "params": [arg.arg for arg in node.args.args],
            "decorators": [ast.unparse(d) for d in node.decorator_list],
            "docstring": ast.get_docstring(node) or "",
            "body_pattern": self.extract_body_pattern(node),
            "template": self.create_function_template(node),
            "mesh_coords": self.calculate_pattern_coordinates(func_source)
        }
        
        return pattern
    
    def extract_class_pattern(self, node: ast.ClassDef, source: str) -> Dict:
        """Extract class pattern for concept mesh"""
        pattern = {
            "name": node.name,
            "type": "class",
            "bases": [ast.unparse(base) for base in node.bases],
            "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
            "docstring": ast.get_docstring(node) or "",
            "template": self.create_class_template(node),
            "mesh_coords": self.calculate_pattern_coordinates(node.name)
        }
        
        return pattern
    
    def extract_error_pattern(self, node: ast.Try, source: str) -> Dict:
        """Extract error handling pattern"""
        handlers = []
        for handler in node.handlers:
            exc_type = ast.unparse(handler.type) if handler.type else "Exception"
            handlers.append({
                "exception": exc_type,
                "var": handler.name,
                "body": len(handler.body)
            })
        
        pattern = {
            "type": "error_handling",
            "handlers": handlers,
            "has_else": len(node.orelse) > 0,
            "has_finally": len(node.finalbody) > 0,
            "template": self.create_error_template(handlers),
            "mesh_coords": self.calculate_pattern_coordinates("error_pattern")
        }
        
        return pattern
    
    def extract_architecture_patterns(self, content: str) -> List[Dict]:
        """Extract high-level architecture patterns"""
        patterns = []
        
        # FastAPI patterns
        if "FastAPI" in content:
            patterns.append({
                "type": "architecture",
                "pattern": "fastapi_server",
                "indicators": ["FastAPI", "@app.", "uvicorn"],
                "template": self.get_fastapi_template()
            })
        
        # MCP Server patterns
        if "MCPServer" in content or "mcp_server" in content:
            patterns.append({
                "type": "architecture", 
                "pattern": "mcp_server",
                "indicators": ["MCPServer", "handle_", "transport"],
                "template": self.get_mcp_template()
            })
        
        # Async patterns
        if "asyncio" in content:
            patterns.append({
                "type": "architecture",
                "pattern": "async_application",
                "indicators": ["asyncio", "async def", "await"],
                "template": self.get_async_template()
            })
        
        return patterns
    
    def create_function_template(self, node: ast.FunctionDef) -> str:
        """Create reusable function template"""
        params = ", ".join(arg.arg for arg in node.args.args)
        
        if self.is_async_function(node):
            template = f"async def {{name}}({params}):\n    '''{{docstring}}'''\n    {{body}}"
        else:
            template = f"def {{name}}({params}):\n    '''{{docstring}}'''\n    {{body}}"
        
        return template
    
    def create_class_template(self, node: ast.ClassDef) -> str:
        """Create reusable class template"""
        bases = ", ".join(ast.unparse(base) for base in node.bases) if node.bases else ""
        
        template = f"""class {{name}}({bases}):
    '''{{docstring}}'''
    
    def __init__(self, {{init_params}}):
        {{init_body}}
    
    {{methods}}"""
        
        return template
    
    def create_error_template(self, handlers: List[Dict]) -> str:
        """Create error handling template"""
        template = "try:\n    {try_body}\n"
        
        for handler in handlers:
            exc = handler["exception"]
            var = f" as {handler['var']}" if handler.get('var') else ""
            template += f"except {exc}{var}:\n    {{handle_{exc.lower()}}}\n"
        
        return template
    
    def get_fastapi_template(self) -> str:
        """FastAPI server template"""
        return '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="{title}")

class {Model}(BaseModel):
    {fields}

@app.post("/{endpoint}")
async def {handler}(data: {Model}):
    """
    {description}
    """
    try:
        result = await process_{endpoint}(data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port={port})
'''
    
    def get_mcp_template(self) -> str:
        """MCP server template"""
        return '''class {Name}MCPServer:
    """
    MCP Server for {purpose}
    """
    
    def __init__(self):
        self.name = "{server_name}"
        self.concept_mesh = ConceptMesh()
        self.setup()
    
    async def handle_{operation}(self, request):
        """Handle {operation} request"""
        # Filter input
        filtered = self.filter_input(request)
        
        # Process with concept mesh
        result = await self.process_{operation}(filtered)
        
        # Filter output
        return self.filter_output(result)
    
    async def process_{operation}(self, data):
        """Core {operation} logic"""
        {logic}
'''
    
    def get_async_template(self) -> str:
        """Async application template"""
        return '''import asyncio

class {Name}AsyncApp:
    """
    Async application for {purpose}
    """
    
    def __init__(self):
        self.tasks = []
        self.running = False
    
    async def start(self):
        """Start the application"""
        self.running = True
        self.tasks = [
            asyncio.create_task(self.{task1}()),
            asyncio.create_task(self.{task2}())
        ]
        await asyncio.gather(*self.tasks)
    
    async def {task1}(self):
        """First async task"""
        while self.running:
            {task1_logic}
            await asyncio.sleep({interval1})
    
    async def stop(self):
        """Graceful shutdown"""
        self.running = False
        await asyncio.gather(*self.tasks, return_exceptions=True)
'''
    
    def is_async_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is async"""
        return isinstance(node, ast.AsyncFunctionDef)
    
    def extract_body_pattern(self, node: ast.FunctionDef) -> str:
        """Extract pattern from function body"""
        # Simplified - in production, do deeper analysis
        if len(node.body) == 0:
            return "empty"
        
        first_stmt = node.body[0]
        if isinstance(first_stmt, ast.Return):
            return "direct_return"
        elif isinstance(first_stmt, ast.Try):
            return "error_first"
        elif isinstance(first_stmt, ast.If):
            return "conditional_first"
        else:
            return "standard"
    
    def calculate_pattern_coordinates(self, content: str) -> List[float]:
        """Calculate 4D coordinates for concept mesh storage"""
        # Hash-based coordinates for LSTM compatibility
        hash_val = hash(content)
        
        return [
            (hash_val % 1000) / 1000,  # ψ: pattern complexity
            ((hash_val >> 10) % 1000) / 1000,  # ε: abstraction level
            ((hash_val >> 20) % 1000) / 1000,  # τ: temporal ordering
            ((hash_val >> 30) % 1000) / 1000,  # φ: domain specificity
        ]
    
    def learn_from_directory(self, directory: Path, recursive: bool = True):
        """Learn patterns from entire directory"""
        print(f"\n📚 Learning from: {directory}")
        print("=" * 60)
        
        all_patterns = {k: [] for k in self.pattern_types.keys()}
        file_count = 0
        
        # Find all Python files
        pattern = "**/*.py" if recursive else "*.py"
        for py_file in directory.glob(pattern):
            # Skip __pycache__ and hidden files
            if "__pycache__" in str(py_file) or py_file.name.startswith('.'):
                continue
            
            print(f"  Analyzing: {py_file.relative_to(directory)}")
            patterns = self.extract_patterns_from_file(py_file)
            
            # Merge patterns
            for pattern_type, pattern_list in patterns.items():
                all_patterns[pattern_type].extend(pattern_list)
            
            file_count += 1
        
        # Save patterns to concept mesh
        self.save_patterns_to_mesh(all_patterns)
        
        print(f"\n✅ Learned from {file_count} files")
        self.print_pattern_summary(all_patterns)
    
    def save_patterns_to_mesh(self, patterns: Dict[str, List[Dict]]):
        """Save extracted patterns to concept mesh"""
        for pattern_type, pattern_list in patterns.items():
            if not pattern_list:
                continue
            
            # Remove duplicates based on template
            unique_patterns = {}
            for pattern in pattern_list:
                template = pattern.get('template', '')
                if template and template not in unique_patterns:
                    unique_patterns[template] = pattern
            
            # Save to concept mesh
            mesh_file = self.code_patterns_dir / f"{pattern_type}.json"
            with open(mesh_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "type": pattern_type,
                    "count": len(unique_patterns),
                    "patterns": list(unique_patterns.values())
                }, f, indent=2)
    
    def print_pattern_summary(self, patterns: Dict[str, List[Dict]]):
        """Print summary of learned patterns"""
        print("\n📊 Pattern Summary:")
        print("-" * 40)
        
        total = 0
        for pattern_type, pattern_list in patterns.items():
            count = len(pattern_list)
            total += count
            if count > 0:
                print(f"  {pattern_type}: {count} patterns")
        
        print("-" * 40)
        print(f"  Total: {total} patterns extracted")
    
    def generate_code_from_pattern(self, pattern_type: str, **kwargs) -> str:
        """Generate code using learned patterns"""
        # Load patterns
        mesh_file = self.code_patterns_dir / f"{pattern_type}.json"
        if not mesh_file.exists():
            return f"# No patterns found for {pattern_type}"
        
        with open(mesh_file, 'r') as f:
            data = json.load(f)
        
        patterns = data.get('patterns', [])
        if not patterns:
            return f"# No patterns available for {pattern_type}"
        
        # Find best matching pattern (simple version)
        # In production, use concept mesh coordinates for similarity
        pattern = patterns[0]  # Just use first for now
        
        # Generate from template
        template = pattern.get('template', '')
        if template:
            # Simple template filling
            for key, value in kwargs.items():
                template = template.replace(f"{{{key}}}", str(value))
            
            return template
        
        return f"# Generated {pattern_type}"

def main():
    """Main function to run rapid code learning"""
    pigpen_root = Path(str(PROJECT_ROOT))
    concept_mesh_dir = pigpen_root / "concept_mesh"
    
    learner = RapidCodeLearner(concept_mesh_dir)
    
    print("🚀 RAPID CODE LEARNING SYSTEM")
    print("Teaching TONKA to code like a badass!")
    print("=" * 60)
    
    # Learn from pigpen (which contains tori)
    learner.learn_from_directory(pigpen_root)
    
    # Test generation
    print("\n🧪 Testing code generation...")
    code = learner.generate_code_from_pattern(
        "api_patterns",
        title="TONKA API",
        Model="CodeRequest",
        fields="code_type: str\n    requirements: List[str]",
        endpoint="generate",
        handler="generate_code",
        description="Generate code based on requirements",
        port=8080
    )
    
    print("\nGenerated code:")
    print("-" * 40)
    print(code)
    
    print("\n✅ TONKA is ready to code!")

if __name__ == "__main__":
    main()
