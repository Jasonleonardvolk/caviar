"""
Python AST to Concept Graph Importer

This module provides functionality to parse Python code into an Abstract Syntax Tree (AST) 
and transform it into a concept graph suitable for visualization in the ALAN IDE.

The concept graph includes:
- Nodes representing functions, classes, variables, etc.
- Edges representing relationships (imports, calls, inheritance)
- Phase dynamics for visualizing semantic properties
- Koopman spectral decomposition for flow analysis
"""

import ast
import os
import re
import json
import hashlib
from typing import Dict, List, Set, Tuple, Any, Optional

# For secret scanning
# We'll make trufflehogsecrets optional 
try:
    import trufflehogsecrets
    TRUFFLEHOG_AVAILABLE = True
except ImportError:
    TRUFFLEHOG_AVAILABLE = False

# Constants
NODE_TYPES = {
    'function': 1,
    'class': 2,
    'module': 3,
    'variable': 4,
    'import': 5,
    'call': 6
}

class SecretScanner:
    """Scans Python code for secrets and sensitive information."""
    
    def __init__(self, rules_path: Optional[str] = None):
        """Initialize the scanner with optional custom rules."""
        self.rules_path = rules_path
        # Default rules would include patterns for API keys, tokens, etc.
        self.default_patterns = [
            r'(?i)api[-_]?key[-_]?[\'"]?[=:]\s*[\'"]([0-9a-zA-Z]{32,64})[\'"]',
            r'(?i)secret[-_]?[\'"]?[=:]\s*[\'"]([0-9a-zA-Z]{16,64})[\'"]',
            r'(?i)password[-_]?[\'"]?[=:]\s*[\'"]([^\'"\s]{8,64})[\'"]',
            r'(?i)access[-_]?token[-_]?[\'"]?[=:]\s*[\'"]([0-9a-zA-Z]{16,64})[\'"]',
        ]
    
    def scan_code(self, code: str) -> List[Dict[str, Any]]:
        """
        Scan code for potential secrets.
        
        Args:
            code: Python code to scan
            
        Returns:
            List of findings with type, value, line number
        """
        findings = []
        
        # In a real implementation, this would use TruffleHog's API
        # For now, we'll simulate with regex
        for pattern in self.default_patterns:
            for match in re.finditer(pattern, code):
                secret_value = match.group(1)
                # Calculate line number
                line_number = code[:match.start()].count('\n') + 1
                
                findings.append({
                    'type': self._detect_secret_type(pattern),
                    'value': secret_value,
                    'line': line_number,
                    'pattern': pattern,
                    'context': self._get_context(code, match.start(), 40)
                })
        
        return findings
    
    def _detect_secret_type(self, pattern: str) -> str:
        """Determine the type of secret based on the pattern."""
        if 'api' in pattern.lower():
            return 'API_KEY'
        elif 'password' in pattern.lower():
            return 'PASSWORD'
        elif 'token' in pattern.lower():
            return 'ACCESS_TOKEN'
        elif 'secret' in pattern.lower():
            return 'SECRET'
        return 'UNKNOWN'
    
    def _get_context(self, code: str, pos: int, context_size: int) -> str:
        """Get surrounding context for a secret."""
        start = max(0, pos - context_size)
        end = min(len(code), pos + context_size)
        return code[start:end].replace('\n', ' ')


class ConceptNode:
    """Represents a node in the concept graph."""
    
    def __init__(self, 
                 node_id: str,
                 label: str,
                 node_type: str,
                 ast_node: Optional[ast.AST] = None,
                 phase: float = 0.0,
                 resonance: float = 0.5):
        """
        Initialize a concept node.
        
        Args:
            node_id: Unique identifier
            label: Display name
            node_type: Type of concept (function, class, etc.)
            ast_node: Original AST node (if applicable)
            phase: Phase value (0.0-1.0) for dynamics visualization
            resonance: Resonance value (0.0-1.0) for visualization
        """
        self.id = node_id
        self.label = label
        self.type = node_type
        self.ast_node = ast_node
        self.phase = phase
        self.resonance = resonance
        self.metadata = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            'id': self.id,
            'label': self.label,
            'type': self.type,
            'phase': self.phase,
            'resonance': self.resonance,
            'metadata': self.metadata
        }
    
    def calculate_phase(self) -> None:
        """
        Calculate phase value based on node properties.
        
        Phase represents semantic meaning in the visualization:
        - Functions related to I/O might have similar phases
        - Functions in the same module might have related phases
        - Classes in inheritance hierarchies have phase relationships
        """
        # This is a simplified algorithm - would be more sophisticated in practice
        
        # Base phase on name - similar names get similar phases
        name_hash = int(hashlib.md5(self.label.encode()).hexdigest(), 16)
        self.phase = (name_hash % 1000) / 1000.0
        
        # Adjust based on type
        type_factors = {
            'function': 0.0,  # No adjustment
            'class': 0.1,     # Slight shift for classes
            'module': 0.2,    # Larger shift for modules
            'variable': 0.3,  # Etc.
            'import': 0.4,
            'call': 0.5
        }
        
        self.phase = (self.phase + type_factors.get(self.type, 0.0)) % 1.0
        
    def calculate_resonance(self) -> None:
        """
        Calculate resonance value based on node properties.
        
        Resonance represents importance or centrality in the visualization:
        - Functions with many calls have higher resonance
        - Classes with many methods have higher resonance
        - Modules with many imports have higher resonance
        """
        # This would be calculated based on the graph structure
        # For now, use a default value
        self.resonance = 0.7


class ConceptEdge:
    """Represents an edge in the concept graph."""
    
    def __init__(self,
                 edge_id: str,
                 source: str,
                 target: str,
                 edge_type: str,
                 weight: float = 0.5):
        """
        Initialize a concept edge.
        
        Args:
            edge_id: Unique identifier
            source: Source node ID
            target: Target node ID
            edge_type: Type of relationship (import, call, etc.)
            weight: Edge weight (0.0-1.0) for visualization
        """
        self.id = edge_id
        self.source = source
        self.target = target
        self.type = edge_type
        self.weight = weight
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            'id': self.id,
            'source': self.source,
            'target': self.target,
            'type': self.type,
            'weight': self.weight,
            'metadata': self.metadata
        }
    
    def calculate_weight(self, nodes_map: Dict[str, ConceptNode]) -> None:
        """
        Calculate edge weight based on properties.
        
        Weight represents strength of relationship in the visualization:
        - More frequent calls have higher weight
        - Inheritance is higher weight than association
        - Direct imports are higher weight than transitive imports
        """
        # In practice, this would use more sophisticated metrics
        # like call frequency, shared references, etc.
        
        # For now, use a simple type-based weighting
        type_weights = {
            'inheritance': 0.9,
            'import': 0.7,
            'call': 0.6,
            'reference': 0.4,
            'association': 0.3
        }
        
        self.weight = type_weights.get(self.type, 0.5)


class ConceptGraphImporter:
    """Converts Python code to a concept graph."""
    
    def __init__(self):
        """Initialize the importer."""
        self.nodes = {}  # id -> ConceptNode
        self.edges = {}  # id -> ConceptEdge
        self.secret_scanner = SecretScanner()
        
    def import_file(self, file_path: str) -> Tuple[Dict[str, ConceptNode], Dict[str, ConceptEdge]]:
        """
        Import a Python file and convert to concept graph.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Tuple of (nodes, edges) dictionaries
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        return self.import_code(code, file_path)
    
    def import_code(self, code: str, source_name: str = "<unknown>") -> Tuple[Dict[str, ConceptNode], Dict[str, ConceptEdge]]:
        """
        Import Python code string and convert to concept graph.
        
        Args:
            code: Python code string
            source_name: Name to identify the source
            
        Returns:
            Tuple of (nodes, edges) dictionaries
        """
        # Parse the code into AST
        try:
            tree = ast.parse(code, filename=source_name)
        except SyntaxError as e:
            print(f"Syntax error in {source_name}: {e}")
            return {}, {}
        
        # Clear previous data
        self.nodes = {}
        self.edges = {}
        
        # Process the AST
        self._process_module(tree, source_name)
        
        # Scan for secrets
        secrets = self.secret_scanner.scan_code(code)
        for secret in secrets:
            secret_id = f"secret_{source_name}_{secret['line']}"
            secret_node = ConceptNode(
                secret_id,
                f"Secret: {secret['type']}",
                'secret',
                None,
                0.9,  # Phase near 1.0 to highlight
                0.8   # High resonance to make it visible
            )
            secret_node.metadata['secret_type'] = secret['type']
            secret_node.metadata['line'] = secret['line']
            secret_node.metadata['context'] = secret['context']
            self.nodes[secret_id] = secret_node
        
        # Calculate phase and resonance for nodes
        for node in self.nodes.values():
            node.calculate_phase()
            node.calculate_resonance()
        
        # Calculate weights for edges
        for edge in self.edges.values():
            edge.calculate_weight(self.nodes)
        
        return self.nodes, self.edges
    
    def import_directory(self, directory_path: str) -> Tuple[Dict[str, ConceptNode], Dict[str, ConceptEdge]]:
        """
        Import all Python files in a directory.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            Tuple of (nodes, edges) dictionaries
        """
        all_nodes = {}
        all_edges = {}
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        nodes, edges = self.import_file(file_path)
                        all_nodes.update(nodes)
                        all_edges.update(edges)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return all_nodes, all_edges
    
    def _process_module(self, module: ast.Module, source_name: str) -> str:
        """Process a module AST node."""
        module_id = f"module_{source_name}"
        module_node = ConceptNode(
            module_id,
            os.path.basename(source_name),
            'module',
            module
        )
        self.nodes[module_id] = module_node
        
        # Process imports
        imports = [node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        for import_node in imports:
            self._process_import(import_node, module_id)
        
        # Process functions
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        for func_node in functions:
            self._process_function(func_node, module_id)
        
        # Process classes
        classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
        for class_node in classes:
            self._process_class(class_node, module_id)
        
        # Process assignments
        assignments = [node for node in module.body if isinstance(node, ast.Assign)]
        for assign_node in assignments:
            self._process_assignment(assign_node, module_id)
        
        return module_id
    
    def _process_import(self, import_node: ast.AST, parent_id: str) -> None:
        """Process an import AST node."""
        if isinstance(import_node, ast.Import):
            for alias in import_node.names:
                import_id = f"import_{parent_id}_{alias.name}"
                import_concept = ConceptNode(
                    import_id,
                    alias.name,
                    'import',
                    import_node
                )
                self.nodes[import_id] = import_concept
                
                # Edge: module imports module
                edge_id = f"edge_{parent_id}_imports_{import_id}"
                edge = ConceptEdge(edge_id, parent_id, import_id, 'import')
                self.edges[edge_id] = edge
                
        elif isinstance(import_node, ast.ImportFrom):
            module_name = import_node.module or ''
            for alias in import_node.names:
                import_id = f"import_{parent_id}_{module_name}_{alias.name}"
                import_concept = ConceptNode(
                    import_id,
                    f"{module_name}.{alias.name}",
                    'import',
                    import_node
                )
                self.nodes[import_id] = import_concept
                
                # Edge: module imports from module
                edge_id = f"edge_{parent_id}_imports_{import_id}"
                edge = ConceptEdge(edge_id, parent_id, import_id, 'import')
                self.edges[edge_id] = edge
    
    def _process_function(self, func_node: ast.FunctionDef, parent_id: str) -> str:
        """Process a function AST node."""
        func_id = f"function_{parent_id}_{func_node.name}"
        func_concept = ConceptNode(
            func_id,
            func_node.name,
            'function',
            func_node
        )
        self.nodes[func_id] = func_concept
        
        # Edge: parent contains function
        edge_id = f"edge_{parent_id}_contains_{func_id}"
        edge = ConceptEdge(edge_id, parent_id, func_id, 'contains')
        self.edges[edge_id] = edge
        
        # Process function body for calls
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                self._process_call(node, func_id)
        
        return func_id
    
    def _process_class(self, class_node: ast.ClassDef, parent_id: str) -> str:
        """Process a class AST node."""
        class_id = f"class_{parent_id}_{class_node.name}"
        class_concept = ConceptNode(
            class_id,
            class_node.name,
            'class',
            class_node
        )
        self.nodes[class_id] = class_concept
        
        # Edge: parent contains class
        edge_id = f"edge_{parent_id}_contains_{class_id}"
        edge = ConceptEdge(edge_id, parent_id, class_id, 'contains')
        self.edges[edge_id] = edge
        
        # Process inheritance
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                base_id = f"class_{parent_id}_{base.id}"
                if base_id in self.nodes:
                    # Edge: class inherits from base
                    edge_id = f"edge_{class_id}_inherits_{base_id}"
                    edge = ConceptEdge(edge_id, class_id, base_id, 'inheritance')
                    self.edges[edge_id] = edge
        
        # Process methods
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_id = self._process_function(node, class_id)
                # Mark as method in metadata
                self.nodes[method_id].metadata['is_method'] = True
        
        return class_id
    
    def _process_call(self, call_node: ast.Call, parent_id: str) -> None:
        """Process a function call AST node."""
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
            call_id = f"call_{parent_id}_{func_name}_{id(call_node)}"
            call_concept = ConceptNode(
                call_id,
                f"call {func_name}()",
                'call',
                call_node
            )
            self.nodes[call_id] = call_concept
            
            # Edge: parent calls function
            edge_id = f"edge_{parent_id}_calls_{call_id}"
            edge = ConceptEdge(edge_id, parent_id, call_id, 'call')
            self.edges[edge_id] = edge
            
    def _process_assignment(self, assign_node: ast.Assign, parent_id: str) -> None:
        """Process a variable assignment AST node."""
        for target in assign_node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                var_id = f"variable_{parent_id}_{var_name}_{id(assign_node)}"
                var_concept = ConceptNode(
                    var_id,
                    var_name,
                    'variable',
                    assign_node
                )
                self.nodes[var_id] = var_concept
                
                # Edge: parent defines variable
                edge_id = f"edge_{parent_id}_defines_{var_id}"
                edge = ConceptEdge(edge_id, parent_id, var_id, 'defines')
                self.edges[edge_id] = edge
    
    def to_json(self) -> str:
        """Convert the concept graph to JSON for visualization."""
        graph = {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges.values()]
        }
        return json.dumps(graph, indent=2)
    
    def save_to_file(self, file_path: str) -> None:
        """Save the concept graph to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


class KoopmanSpectralDecomposer:
    """
    Performs Koopman spectral decomposition on the concept graph.
    
    This is used to identify the dynamical flow patterns in the code,
    which are visualized in the KoopmanOverlay component.
    """
    
    def __init__(self, nodes: Dict[str, ConceptNode], edges: Dict[str, ConceptEdge]):
        """
        Initialize with a concept graph.
        
        Args:
            nodes: Dictionary of concept nodes
            edges: Dictionary of concept edges
        """
        self.nodes = nodes
        self.edges = edges
        
    def compute_koopman_modes(self) -> Dict[str, Any]:
        """
        Compute Koopman modes for the concept graph.
        
        Returns:
            Dictionary with eigenvalues, eigenvectors, and flow data
        """
        # In a real implementation, this would use proper numerical methods
        # like EDMD (Extended Dynamic Mode Decomposition)
        #
        # For this implementation, we'll return mock data that's compatible
        # with the KoopmanOverlay component
        
        # Mock flow vectors for each node
        flow_vectors = {}
        for node_id, node in self.nodes.items():
            # Use phase as angle
            angle = node.phase * 2 * 3.14159
            # Use resonance as magnitude
            magnitude = node.resonance * 2.0
            
            flow_vectors[node_id] = {
                'dx': magnitude * 0.7 * 0.1 * magnitude * 0.7 * magnitude * 0.7 * 0.1 * magnitude * 0.7,
                'dy': magnitude * 0.7 * 0.1 * magnitude * 0.7 * magnitude * 0.7 * 0.1 * magnitude * 0.7,
                'magnitude': magnitude
            }
        
        return {
            'eigenvalues': [0.9 + 0.1j, 0.8 - 0.2j, 0.7, 0.6, 0.5],
            'eigenvectors': [
                [0.1, 0.2, 0.3], 
                [0.3, 0.2, 0.1],
                [0.2, 0.2, 0.2]
            ],
            'flow_vectors': flow_vectors
        }


def generate_layout(nodes: Dict[str, ConceptNode]) -> Dict[str, Tuple[float, float]]:
    """
    Generate an initial 2D layout for the nodes.
    
    This would be replaced with a more sophisticated layout algorithm
    like force-directed, but provides a starting point for visualization.
    
    Args:
        nodes: Dictionary of concept nodes
        
    Returns:
        Dictionary mapping node IDs to (x, y) coordinates
    """
    import math
    import random
    
    layout = {}
    
    # Group nodes by type
    nodes_by_type = {}
    for node_id, node in nodes.items():
        if node.type not in nodes_by_type:
            nodes_by_type[node.type] = []
        nodes_by_type[node.type].append(node_id)
    
    # Layout nodes in concentric circles by type
    canvas_width = 2000
    canvas_height = 2000
    center_x = canvas_width / 2
    center_y = canvas_height / 2
    
    for i, (node_type, node_ids) in enumerate(nodes_by_type.items()):
        radius = 150 * (i + 1)
        for j, node_id in enumerate(node_ids):
            angle = (2 * math.pi * j) / len(node_ids)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Add some randomness
            x += random.uniform(-20, 20)
            y += random.uniform(-20, 20)
            
            layout[node_id] = (x, y)
    
    return layout


def main():
    """Demo function to test the importer."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python python_to_concept_graph.py <python_file_or_directory>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    importer = ConceptGraphImporter()
    
    if os.path.isfile(path):
        nodes, edges = importer.import_file(path)
    elif os.path.isdir(path):
        nodes, edges = importer.import_directory(path)
    else:
        print(f"Path not found: {path}")
        sys.exit(1)
    
    print(f"Imported {len(nodes)} nodes and {len(edges)} edges")
    
    # Generate layout
    layout = generate_layout(nodes)
    
    # Add coordinates to nodes
    for node_id, (x, y) in layout.items():
        nodes[node_id].metadata['x'] = x
        nodes[node_id].metadata['y'] = y
    
    # Compute Koopman modes
    decomposer = KoopmanSpectralDecomposer(nodes, edges)
    koopman_data = decomposer.compute_koopman_modes()
    
    # Save to file
    output_file = "concept_graph.json"
    importer.save_to_file(output_file)
    print(f"Saved concept graph to {output_file}")


if __name__ == "__main__":
    main()
