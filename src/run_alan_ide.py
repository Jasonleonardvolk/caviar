"""
ALAN IDE Phase 3 Starter Script

This script demonstrates the functionality of the ALAN IDE Phase 3 implementation,
integrating Python AST import, concept graph visualization, and the Project Vault.

Usage:
    python run_alan_ide.py [python_file_or_directory]
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Import our components
try:
    from python_to_concept_graph import ConceptGraphImporter
    from project_vault_service import VaultService, start_api_server
    from import_wizard import ImportWizard
    DIRECT_IMPORT = True
except ImportError:
    DIRECT_IMPORT = False

# Optional rich UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    required_packages = [
        "cryptography",  # For vault encryption
        "flask",         # For vault API
        "rich"           # For UI (optional)
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        if console:
            console.print(Panel(
                f"Missing required packages: [bold red]{', '.join(missing_packages)}[/bold red]\n\n"
                f"Install them with: [bold]pip install {' '.join(missing_packages)}[/bold]",
                title="Dependency Check Failed",
                border_style="red"
            ))
        else:
            print("Missing required packages:", ", ".join(missing_packages))
            print(f"Install them with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_services() -> Dict[str, bool]:
    """Check if ALAN IDE services are available."""
    services = {
        "vault_service": False,
        "import_wizard": False,
        "concept_graph": False
    }
    
    if not DIRECT_IMPORT:
        # Check if the Python files exist
        services["vault_service"] = os.path.exists("src/project_vault_service.py")
        services["import_wizard"] = os.path.exists("src/import_wizard.py")
        services["concept_graph"] = os.path.exists("src/python_to_concept_graph.py")
    else:
        # We already imported them successfully
        services["vault_service"] = True
        services["import_wizard"] = True
        services["concept_graph"] = True
    
    return services

def start_vault_service(host: str = '127.0.0.1', port: int = 5000) -> subprocess.Popen:
    """Start the vault service API server."""
    if DIRECT_IMPORT:
        # Use the imported function in a subprocess
        import threading
        threading.Thread(target=start_api_server, args=(host, port), daemon=True).start()
        return None
    else:
        # Start as a separate process
        vault_path = os.path.join("src", "project_vault_service.py")
        if not os.path.exists(vault_path):
            raise FileNotFoundError(f"Vault service not found at {vault_path}")
        
        return subprocess.Popen(
            [sys.executable, vault_path, 'server', '--host', host, '--port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

def run_import_wizard(project_path: Optional[str] = None) -> Dict[str, Any]:
    """Run the import wizard on a Python project."""
    output_path = os.path.join("output", "import_results.json")
    graph_path = os.path.join("output", "concept_graph.json")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if DIRECT_IMPORT:
        # Use the imported class
        wizard = ImportWizard()
        result = wizard.start(project_path)
        
        # Save results and graph
        wizard.save_results(output_path, result)
        wizard.generate_concept_graph_json(graph_path)
        
        return result
    else:
        # Run as a subprocess
        wizard_path = os.path.join("src", "import_wizard.py")
        if not os.path.exists(wizard_path):
            raise FileNotFoundError(f"Import wizard not found at {wizard_path}")
        
        cmd = [sys.executable, wizard_path]
        if project_path:
            cmd.append(project_path)
        cmd.extend(['--output', output_path, '--graph', graph_path])
        
        subprocess.run(cmd, check=True)
        
        # Load the results
        with open(output_path, 'r') as f:
            return json.load(f)

def start_concept_canvas_viewer(concept_graph_path: str) -> subprocess.Popen:
    """
    Start a simple viewer for the concept graph.
    
    In a full implementation, this would start a web server to render
    the ConceptFieldCanvas with the concept graph data. For this demo,
    we just print the concept graph stats.
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(concept_graph_path), exist_ok=True)
    
    if not os.path.exists(concept_graph_path):
        raise FileNotFoundError(f"Concept graph not found at {concept_graph_path}")
    
    with open(concept_graph_path, 'r') as f:
        graph_data = json.load(f)
    
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    if console:
        console.print(Panel(
            f"Concept Graph Statistics\n\n"
            f"Nodes: [bold green]{len(nodes)}[/bold green]\n"
            f"Edges: [bold blue]{len(edges)}[/bold blue]\n",
            title="Concept Graph Viewer",
            border_style="green"
        ))
        
        # Show node types
        node_types = {}
        for node in nodes:
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        table = Table(title="Node Types")
        table.add_column("Type", style="bold")
        table.add_column("Count", style="cyan")
        
        for node_type, count in node_types.items():
            table.add_row(node_type, str(count))
        
        console.print(table)
        
        # Show a sample of nodes
        console.print(Panel(
            Syntax(json.dumps(nodes[:5], indent=2), "json", theme="monokai"),
            title="Sample Nodes",
            border_style="blue"
        ))
    else:
        print(f"Concept Graph Statistics:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")
        
        # Show node types
        node_types = {}
        for node in nodes:
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print("Node Types:")
        for node_type, count in node_types.items():
            print(f"  {node_type}: {count}")
    
    return None  # No subprocess to return in this simplified version

def main():
    """Main function to run the ALAN IDE demo."""
    parser = argparse.ArgumentParser(description='ALAN IDE Phase 3 Demo')
    parser.add_argument('path', nargs='?', default='src', help='Path to Python project (default: src)')
    parser.add_argument('--vault-port', type=int, default=5000, help='Port for vault service')
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check services
    services = check_services()
    if not all(services.values()):
        missing = [name for name, available in services.items() if not available]
        if console:
            console.print(Panel(
                f"Missing services: [bold red]{', '.join(missing)}[/bold red]",
                title="Service Check Failed",
                border_style="red"
            ))
        else:
            print("Missing services:", ", ".join(missing))
        sys.exit(1)
    
    if console:
        console.print(Panel(
            "Welcome to ALAN IDE Phase 3 Demo!",
            title="ALAN IDE",
            subtitle="Phase 3 - Sprint 1",
            border_style="bold green"
        ))
    else:
        print("===== ALAN IDE Phase 3 Demo =====")
    
    try:
        # Step 1: Start the vault service
        if console:
            console.print("Starting vault service...")
        else:
            print("Starting vault service...")
        
        vault_proc = start_vault_service(port=args.vault_port)
        
        # Step 2: Run the import wizard
        if console:
            console.print("Starting import wizard...")
        else:
            print("Starting import wizard...")
        
        import_results = run_import_wizard(args.path)
        
        # Step 3: View the concept graph
        if console:
            console.print("Viewing concept graph...")
        else:
            print("Viewing concept graph...")
        
        graph_path = os.path.join("output", "concept_graph.json")
        start_concept_canvas_viewer(graph_path)
        
        if console:
            console.print(Panel(
                f"ALAN IDE Phase 3 Demo completed successfully!\n\n"
                f"View the import results in [bold]output/import_results.json[/bold]\n"
                f"View the concept graph in [bold]output/concept_graph.json[/bold]",
                title="Success",
                border_style="green"
            ))
        else:
            print("\nALAN IDE Phase 3 Demo completed successfully!")
            print("View the import results in output/import_results.json")
            print("View the concept graph in output/concept_graph.json")
    
    except Exception as e:
        if console:
            console.print(Panel(
                f"Error: [bold red]{e}[/bold red]",
                title="Error",
                border_style="red"
            ))
        else:
            print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        if vault_proc:
            vault_proc.terminate()


if __name__ == '__main__':
    main()
