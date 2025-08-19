"""
Import Wizard with Secret Scanning and Vault Migration

This module provides a wizard-style interface for importing Python projects,
scanning for secrets, and migrating them to the Project Vault.

It integrates:
- Python AST to Concept Graph conversion
- Secret detection and scanning
- Project Vault integration for secure storage
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union

# Import our components
try:
    from python_to_concept_graph import ConceptGraphImporter, SecretScanner
    from project_vault_service import VaultService
except ImportError as e:
    print(f"Error importing components: {e}")
    print("Make sure you've installed required dependencies.")
    sys.exit(1)

# For terminal UI (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ImportWizard:
    """
    Wizard for importing Python projects with secret scanning and vault migration.
    
    This provides a step-by-step process for:
    1. Selecting/configuring the Python project to import
    2. Scanning for secrets and sensitive information
    3. Reviewing detected secrets
    4. Migrating secrets to the vault
    5. Running the concept graph import
    """
    
    def __init__(self):
        """Initialize the wizard."""
        self.importer = ConceptGraphImporter()
        self.vault = VaultService()
        
        # Console for UI
        self.console = Console() if RICH_AVAILABLE else None
        
        # Wizard state
        self.project_path = None
        self.detected_secrets = []
        self.vault_migration_map = {}  # original_value -> vault_key
        self.excluded_patterns = []
        self.concept_nodes = {}
        self.concept_edges = {}
    
    def start(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Start the import wizard.
        
        Args:
            project_path: Optional path to Python project
            
        Returns:
            Dictionary with wizard results
        """
        # Step 1: Project Selection
        self.project_path = self._step_project_selection(project_path)
        
        # Step 2: Secret Scanning
        self.detected_secrets = self._step_secret_scanning()
        
        # Step 3: Secret Review
        self.vault_migration_map = self._step_secret_review()
        
        # Step 4: Vault Migration
        migrated_secrets = self._step_vault_migration()
        
        # Step 5: Concept Graph Import
        self.concept_nodes, self.concept_edges = self._step_concept_import()
        
        # Step 6: Summary
        return self._step_summary(migrated_secrets)
    
    def _step_project_selection(self, project_path: Optional[str] = None) -> str:
        """
        Step 1: Project Selection
        
        Args:
            project_path: Optional path to Python project
            
        Returns:
            Validated project path
        """
        if self.console:
            self.console.print(Panel("Step 1: Project Selection", style="bold blue"))
        else:
            print("\n=== Step 1: Project Selection ===\n")
        
        # Use provided path or prompt
        if project_path:
            path = os.path.abspath(project_path)
        else:
            if self.console:
                path = input("Enter path to Python project: ")
            else:
                path = input("Enter path to Python project: ")
            path = os.path.abspath(path)
        
        # Validate project path
        if not os.path.exists(path):
            raise ValueError(f"Project path does not exist: {path}")
        
        if os.path.isfile(path):
            if not path.endswith('.py'):
                raise ValueError(f"File is not a Python file: {path}")
        else:  # Directory
            # Check if it contains Python files
            py_files = list(glob.glob(os.path.join(path, "**/*.py"), recursive=True))
            if not py_files:
                raise ValueError(f"No Python files found in project: {path}")
        
        if self.console:
            self.console.print(f"Selected project: [bold green]{path}[/bold green]")
        else:
            print(f"Selected project: {path}")
        
        return path
    
    def _step_secret_scanning(self) -> List[Dict[str, Any]]:
        """
        Step 2: Secret Scanning
        
        Scans the project for secrets and sensitive information.
        
        Returns:
            List of detected secrets
        """
        if self.console:
            self.console.print(Panel("Step 2: Secret Scanning", style="bold blue"))
        else:
            print("\n=== Step 2: Secret Scanning ===\n")
        
        all_secrets = []
        
        # Use a progress indicator
        if self.console and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Scanning for secrets...", total=None)
                
                # Scan the project
                if os.path.isfile(self.project_path):
                    # Single file
                    all_secrets = self._scan_file(self.project_path)
                else:
                    # Directory
                    py_files = list(glob.glob(os.path.join(self.project_path, "**/*.py"), recursive=True))
                    progress.update(task, total=len(py_files))
                    
                    for file_path in py_files:
                        all_secrets.extend(self._scan_file(file_path))
                        progress.update(task, advance=1)
                
                progress.update(task, completed=True)
        else:
            # No rich UI, just print status
            print("Scanning for secrets...")
            
            # Scan the project
            if os.path.isfile(self.project_path):
                # Single file
                all_secrets = self._scan_file(self.project_path)
            else:
                # Directory
                py_files = list(glob.glob(os.path.join(self.project_path, "**/*.py"), recursive=True))
                
                for i, file_path in enumerate(py_files):
                    print(f"Scanning {i+1}/{len(py_files)}: {os.path.basename(file_path)}")
                    all_secrets.extend(self._scan_file(file_path))
            
            print("Scanning complete.")
        
        # Print summary
        if self.console:
            self.console.print(f"Found [bold red]{len(all_secrets)}[/bold red] potential secrets.")
        else:
            print(f"Found {len(all_secrets)} potential secrets.")
        
        return all_secrets
    
    def _scan_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Scan a single file for secrets.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of detected secrets
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            scanner = SecretScanner()
            findings = scanner.scan_code(code)
            
            # Add file path to findings
            for finding in findings:
                finding['file_path'] = file_path
            
            return findings
        except Exception as e:
            if self.console:
                self.console.print(f"Error scanning {file_path}: {e}", style="bold red")
            else:
                print(f"Error scanning {file_path}: {e}")
            return []
    
    def _step_secret_review(self) -> Dict[str, str]:
        """
        Step 3: Secret Review
        
        Review detected secrets and decide which to migrate to the vault.
        
        Returns:
            Dictionary mapping secret values to vault keys
        """
        if self.console:
            self.console.print(Panel("Step 3: Secret Review", style="bold blue"))
        else:
            print("\n=== Step 3: Secret Review ===\n")
        
        if not self.detected_secrets:
            if self.console:
                self.console.print("No secrets detected. Skipping review.")
            else:
                print("No secrets detected. Skipping review.")
            return {}
        
        # Display detected secrets
        if self.console and RICH_AVAILABLE:
            table = Table(title=f"Detected Secrets ({len(self.detected_secrets)})")
            table.add_column("ID", style="dim")
            table.add_column("Type", style="bold")
            table.add_column("File", style="blue")
            table.add_column("Line", style="cyan")
            table.add_column("Context", style="green")
            
            for i, secret in enumerate(self.detected_secrets):
                file_name = os.path.basename(secret['file_path'])
                table.add_row(
                    str(i+1),
                    secret['type'],
                    file_name,
                    str(secret['line']),
                    secret['context']
                )
            
            self.console.print(table)
        else:
            print(f"Detected Secrets ({len(self.detected_secrets)}):")
            for i, secret in enumerate(self.detected_secrets):
                file_name = os.path.basename(secret['file_path'])
                print(f"{i+1}. {secret['type']} in {file_name} (line {secret['line']}): {secret['context']}")
            print()
        
        # Allow user to select secrets to migrate
        vault_migration_map = {}
        
        if self.console:
            self.console.print("Select secrets to migrate to vault (comma-separated IDs, 'all', or 'none'):")
            selection = input("> ")
        else:
            print("Select secrets to migrate to vault (comma-separated IDs, 'all', or 'none'):")
            selection = input("> ")
        
        # Process selection
        selected_indices = []
        if selection.lower() == 'all':
            selected_indices = list(range(len(self.detected_secrets)))
        elif selection.lower() != 'none':
            try:
                parts = selection.split(',')
                selected_indices = [int(p.strip()) - 1 for p in parts if p.strip()]
                selected_indices = [i for i in selected_indices 
                                   if 0 <= i < len(self.detected_secrets)]
            except ValueError:
                if self.console:
                    self.console.print("Invalid selection. Using none.", style="bold red")
                else:
                    print("Invalid selection. Using none.")
                selected_indices = []
        
        # For each selected secret, create a vault key
        for idx in selected_indices:
            secret = self.detected_secrets[idx]
            
            # Generate a vault key
            file_name = os.path.basename(secret['file_path']).replace('.py', '')
            vault_key = f"{file_name.upper()}_{secret['type']}"
            
            # Check if the key already exists
            existing_keys = self.vault.list_keys()
            if vault_key in existing_keys:
                i = 1
                while f"{vault_key}_{i}" in existing_keys:
                    i += 1
                vault_key = f"{vault_key}_{i}"
            
            # Add to migration map
            vault_migration_map[secret['value']] = vault_key
        
        # Summary
        if self.console:
            self.console.print(f"Selected [bold green]{len(vault_migration_map)}[/bold green] secrets to migrate.")
        else:
            print(f"Selected {len(vault_migration_map)} secrets to migrate.")
        
        return vault_migration_map
    
    def _step_vault_migration(self) -> List[str]:
        """
        Step 4: Vault Migration
        
        Migrate selected secrets to the vault.
        
        Returns:
            List of migrated secret keys
        """
        if self.console:
            self.console.print(Panel("Step 4: Vault Migration", style="bold blue"))
        else:
            print("\n=== Step 4: Vault Migration ===\n")
        
        migrated_keys = []
        
        if not self.vault_migration_map:
            if self.console:
                self.console.print("No secrets selected for migration. Skipping.")
            else:
                print("No secrets selected for migration. Skipping.")
            return migrated_keys
        
        # Migrate each secret
        for secret_value, vault_key in self.vault_migration_map.items():
            # Find the secret in detected_secrets to get metadata
            metadata = {}
            for secret in self.detected_secrets:
                if secret['value'] == secret_value:
                    metadata = {
                        'type': secret['type'],
                        'file_path': secret['file_path'],
                        'line': secret['line'],
                        'context': secret['context'],
                        'import_date': 'now'  # This should use a proper timestamp
                    }
                    break
            
            # Store in vault
            success = self.vault.put(vault_key, secret_value, metadata)
            
            if success:
                migrated_keys.append(vault_key)
                if self.console:
                    self.console.print(f"Migrated secret to [bold green]{vault_key}[/bold green]")
                else:
                    print(f"Migrated secret to {vault_key}")
            else:
                if self.console:
                    self.console.print(f"Failed to migrate secret to [bold red]{vault_key}[/bold red]")
                else:
                    print(f"Failed to migrate secret to {vault_key}")
        
        # Summary
        if self.console:
            self.console.print(f"Successfully migrated [bold green]{len(migrated_keys)}/{len(self.vault_migration_map)}[/bold green] secrets.")
        else:
            print(f"Successfully migrated {len(migrated_keys)}/{len(self.vault_migration_map)} secrets.")
        
        return migrated_keys
    
    def _step_concept_import(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Step 5: Concept Graph Import
        
        Import the project into a concept graph.
        
        Returns:
            Tuple of (nodes, edges) dictionaries
        """
        if self.console:
            self.console.print(Panel("Step 5: Concept Graph Import", style="bold blue"))
        else:
            print("\n=== Step 5: Concept Graph Import ===\n")
        
        # Use a progress indicator
        if self.console and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Importing project...", total=None)
                
                # Import the project
                if os.path.isfile(self.project_path):
                    # Single file
                    nodes, edges = self.importer.import_file(self.project_path)
                else:
                    # Directory
                    nodes, edges = self.importer.import_directory(self.project_path)
                
                progress.update(task, completed=True)
        else:
            # No rich UI, just print status
            print("Importing project...")
            
            # Import the project
            if os.path.isfile(self.project_path):
                # Single file
                nodes, edges = self.importer.import_file(self.project_path)
            else:
                # Directory
                nodes, edges = self.importer.import_directory(self.project_path)
            
            print("Import complete.")
        
        # Store the nodes and edges
        self.concept_nodes = nodes
        self.concept_edges = edges
        
        # Print summary
        if self.console:
            self.console.print(f"Imported [bold green]{len(nodes)}[/bold green] nodes and [bold green]{len(edges)}[/bold green] edges.")
        else:
            print(f"Imported {len(nodes)} nodes and {len(edges)} edges.")
        
        return nodes, edges
    
    def _step_summary(self, migrated_secrets: List[str]) -> Dict[str, Any]:
        """
        Step 6: Summary
        
        Summarize the import and provide next steps.
        
        Args:
            migrated_secrets: List of migrated secret keys
            
        Returns:
            Dictionary with wizard results
        """
        if self.console:
            self.console.print(Panel("Step 6: Summary", style="bold blue"))
        else:
            print("\n=== Step 6: Summary ===\n")
        
        # Calculate replacement stats
        secret_files = set()
        for secret in self.detected_secrets:
            secret_files.add(secret['file_path'])
        
        # Create result summary
        result = {
            'project_path': self.project_path,
            'detected_secrets': len(self.detected_secrets),
            'migrated_secrets': len(migrated_secrets),
            'affected_files': len(secret_files),
            'concept_nodes': len(self.concept_nodes),
            'concept_edges': len(self.concept_edges),
            'vault_keys': migrated_secrets,
            'replacement_map': self.vault_migration_map
        }
        
        # Display summary
        if self.console:
            table = Table(title="Import Summary")
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="cyan")
            
            table.add_row("Project Path", self.project_path)
            table.add_row("Detected Secrets", str(result['detected_secrets']))
            table.add_row("Migrated Secrets", str(result['migrated_secrets']))
            table.add_row("Affected Files", str(result['affected_files']))
            table.add_row("Concept Nodes", str(result['concept_nodes']))
            table.add_row("Concept Edges", str(result['concept_edges']))
            
            self.console.print(table)
            
            if migrated_secrets:
                vault_table = Table(title="Vault Migration")
                vault_table.add_column("Vault Key", style="bold green")
                
                for key in migrated_secrets:
                    vault_table.add_row(key)
                
                self.console.print(vault_table)
        else:
            print("Import Summary:")
            print(f"  Project Path: {self.project_path}")
            print(f"  Detected Secrets: {result['detected_secrets']}")
            print(f"  Migrated Secrets: {result['migrated_secrets']}")
            print(f"  Affected Files: {result['affected_files']}")
            print(f"  Concept Nodes: {result['concept_nodes']}")
            print(f"  Concept Edges: {result['concept_edges']}")
            
            if migrated_secrets:
                print("\nVault Migration:")
                for key in migrated_secrets:
                    print(f"  {key}")
        
        return result
    
    def save_results(self, output_path: str, result: Dict[str, Any]) -> None:
        """
        Save import results to a file.
        
        Args:
            output_path: Path to save results
            result: Import result dictionary
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            if self.console:
                self.console.print(f"Results saved to [bold green]{output_path}[/bold green]")
            else:
                print(f"Results saved to {output_path}")
        except Exception as e:
            if self.console:
                self.console.print(f"Error saving results: {e}", style="bold red")
            else:
                print(f"Error saving results: {e}")
    
    def generate_concept_graph_json(self, output_path: str) -> None:
        """
        Generate a JSON file with the concept graph.
        
        Args:
            output_path: Path to save the concept graph JSON
        """
        # Make sure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add layout coordinates to nodes
        import random
        import math
        
        canvas_width = 2000
        canvas_height = 2000
        center_x = canvas_width / 2
        center_y = canvas_height / 2
        
        # Group nodes by type for layout
        nodes_by_type = {}
        for node_id, node in self.concept_nodes.items():
            if node.type not in nodes_by_type:
                nodes_by_type[node.type] = []
            nodes_by_type[node.type].append(node_id)
        
        # Layout nodes in concentric circles by type
        for i, (node_type, node_ids) in enumerate(nodes_by_type.items()):
            radius = 150 * (i + 1)
            for j, node_id in enumerate(node_ids):
                angle = (2 * math.pi * j) / max(1, len(node_ids))
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                
                # Add some randomness
                x += random.uniform(-20, 20)
                y += random.uniform(-20, 20)
                
                self.concept_nodes[node_id].metadata['x'] = x
                self.concept_nodes[node_id].metadata['y'] = y
        
        # Convert to JSON and save
        graph = {
            'nodes': [node.to_dict() for node in self.concept_nodes.values()],
            'edges': [edge.to_dict() for edge in self.concept_edges.values()]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2)
        
        if self.console:
            self.console.print(f"Concept graph saved to [bold green]{output_path}[/bold green]")
        else:
            print(f"Concept graph saved to {output_path}")


def main():
    """Command-line interface for the import wizard."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ALAN IDE Import Wizard')
    parser.add_argument('path', nargs='?', help='Path to Python project')
    parser.add_argument('--output', '-o', help='Output path for import results')
    parser.add_argument('--graph', '-g', help='Output path for concept graph JSON')
    
    args = parser.parse_args()
    
    try:
        # Start the wizard
        wizard = ImportWizard()
        result = wizard.start(args.path)
        
        # Save results if requested
        if args.output:
            wizard.save_results(args.output, result)
        
        # Generate concept graph JSON if requested
        if args.graph:
            wizard.generate_concept_graph_json(args.graph)
    except Exception as e:
        if RICH_AVAILABLE:
            console = Console()
            console.print(f"Error: {e}", style="bold red")
        else:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
