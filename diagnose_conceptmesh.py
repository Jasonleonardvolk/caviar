#!/usr/bin/env python3
"""
ConceptMesh Diagnostic Tool
Analyzes ConceptMesh usage across the codebase
"""

import os
import re
import ast
from pathlib import Path
from collections import defaultdict

class ConceptMeshAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze ConceptMesh usage"""
    
    def __init__(self):
        self.imports = []
        self.instantiations = []
        self.class_definitions = []
        
    def visit_Import(self, node):
        for alias in node.names:
            if 'concept_mesh' in alias.name.lower():
                self.imports.append({
                    'type': 'import',
                    'module': alias.name,
                    'alias': alias.asname,
                    'line': node.lineno
                })
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module and 'concept_mesh' in node.module.lower():
            for alias in node.names:
                self.imports.append({
                    'type': 'from_import',
                    'module': node.module,
                    'name': alias.name,
                    'alias': alias.asname,
                    'line': node.lineno
                })
        self.generic_visit(node)
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'ConceptMesh':
            # Analyze the arguments
            args_info = {
                'positional': len(node.args),
                'keywords': {kw.arg: ast.unparse(kw.value) if hasattr(ast, 'unparse') else str(kw.value) 
                            for kw in node.keywords},
                'line': node.lineno
            }
            self.instantiations.append(args_info)
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        if node.name == 'ConceptMesh':
            # Get the __init__ method signature
            init_method = None
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    init_method = item
                    break
            
            if init_method:
                params = []
                for arg in init_method.args.args[1:]:  # Skip 'self'
                    params.append(arg.arg)
                
                self.class_definitions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'init_params': params
                })
        self.generic_visit(node)

def analyze_file(filepath):
    """Analyze a Python file for ConceptMesh usage"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'ConceptMesh' not in content:
            return None
        
        tree = ast.parse(content)
        analyzer = ConceptMeshAnalyzer()
        analyzer.visit(tree)
        
        return {
            'imports': analyzer.imports,
            'instantiations': analyzer.instantiations,
            'class_definitions': analyzer.class_definitions,
            'content': content
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    """Main diagnostic function"""
    print("🔍 ConceptMesh Diagnostic Tool")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    results = defaultdict(list)
    
    # Find all Python files
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in {'.venv', '__pycache__', 'venv', '.git', 'node_modules'}]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                analysis = analyze_file(filepath)
                
                if analysis and 'error' not in analysis:
                    if analysis['imports'] or analysis['instantiations'] or analysis['class_definitions']:
                        results[filepath] = analysis

    # Report findings
    print(f"\n📊 Found {len(results)} files with ConceptMesh references\n")
    
    # Class definitions
    print("📦 ConceptMesh Class Definitions:")
    print("-" * 60)
    for filepath, data in results.items():
        if data['class_definitions']:
            for class_def in data['class_definitions']:
                print(f"\n{filepath}:{class_def['line']}")
                print(f"  Parameters: {', '.join(class_def['init_params'])}")
    
    # Imports
    print("\n\n📥 ConceptMesh Imports:")
    print("-" * 60)
    import_summary = defaultdict(int)
    for filepath, data in results.items():
        if data['imports']:
            print(f"\n{filepath}:")
            for imp in data['imports']:
                print(f"  Line {imp['line']}: {imp['type']} {imp['module']}")
                if imp['name'] if 'name' in imp else None:
                    print(f"    Importing: {imp['name']}")
                import_summary[imp['module']] += 1
    
    # Instantiations
    print("\n\n🔧 ConceptMesh Instantiations:")
    print("-" * 60)
    problem_instantiations = []
    for filepath, data in results.items():
        if data['instantiations']:
            print(f"\n{filepath}:")
            for inst in data['instantiations']:
                print(f"  Line {inst['line']}:")
                print(f"    Positional args: {inst['positional']}")
                if inst['keywords']:
                    print(f"    Keyword args: {inst['keywords']}")
                    problem_instantiations.append((filepath, inst['line'], inst['keywords']))
                else:
                    print(f"    ✅ No keyword arguments (correct)")
    
    # Summary and recommendations
    print("\n\n📋 SUMMARY AND RECOMMENDATIONS:")
    print("=" * 60)
    
    if problem_instantiations:
        print(f"\n⚠️  Found {len(problem_instantiations)} problematic instantiations:")
        for filepath, line, kwargs in problem_instantiations:
            print(f"  - {filepath}:{line} uses keywords: {list(kwargs.keys())}")
        print("\n💡 Recommendation: Run 'python fix_conceptmesh_init.py' to fix these issues")
    else:
        print("\n✅ No problematic ConceptMesh instantiations found!")
    
    print(f"\n📊 Import Summary:")
    for module, count in import_summary.items():
        print(f"  - {module}: {count} imports")
    
    # Save detailed report
    report_path = project_root / "CONCEPTMESH_DIAGNOSTIC_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# ConceptMesh Diagnostic Report\n\n")
        f.write(f"Generated: {Path(__file__).stat().st_mtime}\n\n")
        
        f.write("## Summary\n")
        f.write(f"- Files analyzed: {len(results)}\n")
        f.write(f"- Problem instantiations: {len(problem_instantiations)}\n\n")
        
        if problem_instantiations:
            f.write("## Problematic Instantiations\n\n")
            for filepath, line, kwargs in problem_instantiations:
                f.write(f"### {filepath}:{line}\n")
                f.write(f"Keywords used: `{kwargs}`\n\n")
        
        f.write("\n## Full Analysis\n\n")
        for filepath, data in results.items():
            f.write(f"### {filepath}\n")
            if data['imports']:
                f.write("**Imports:**\n")
                for imp in data['imports']:
                    f.write(f"- Line {imp['line']}: {imp}\n")
            if data['instantiations']:
                f.write("\n**Instantiations:**\n")
                for inst in data['instantiations']:
                    f.write(f"- Line {inst['line']}: {inst}\n")
            if data['class_definitions']:
                f.write("\n**Class Definitions:**\n")
                for cls in data['class_definitions']:
                    f.write(f"- Line {cls['line']}: {cls}\n")
            f.write("\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
