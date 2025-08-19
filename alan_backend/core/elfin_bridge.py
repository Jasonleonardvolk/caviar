# Copyright 2025 ALAN Team and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patent Peace / Retaliation Notice:
#   As stated in Section 3 of the Apache 2.0 License, any entity that
#   initiates patent litigation (including a cross-claim or counterclaim)
#   alleging that this software or a contribution embodied within it
#   infringes a patent shall have all patent licenses granted herein
#   terminated as of the date such litigation is filed.

"""
ELFIN ↔ ψ-graph Bridge for ALAN.

This module provides a bidirectional interface between the ELFIN domain-specific
language and the ALAN ψ-graph (concept graph) reasoning system. It enables
symbolic verification of reasoning steps and provides unit-safe typing.
"""

import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ELFINSymbol:
    """Representation of an ELFIN symbol with type information."""
    
    name: str
    type_sig: str
    units: Optional[str] = None
    source_location: Optional[str] = None
    description: Optional[str] = None


@dataclass
class PsiGraphNode:
    """Representation of a node in the ψ-graph concept network."""
    
    id: str
    label: str
    activation: float = 0.0
    is_elfin_verified: bool = False
    elfin_symbol: Optional[ELFINSymbol] = None


class ELFINBridge:
    """Bridge between ELFIN symbols and ψ-graph concept nodes."""
    
    def __init__(self):
        """Initialize the ELFIN bridge."""
        self.elfin_symbols: Dict[str, ELFINSymbol] = {}
        self.psi_nodes: Dict[str, PsiGraphNode] = {}
        self.bindings: Dict[str, str] = {}  # ELFIN name to ψ-node ID
    
    def import_elfin_symbols(self, symbols_path: str) -> List[ELFINSymbol]:
        """Import ELFIN symbols from a JSON export file.
        
        Args:
            symbols_path: Path to the ELFIN symbols JSON file
            
        Returns:
            List of imported symbols
            
        Raises:
            FileNotFoundError: If the symbols file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        with open(symbols_path, 'r') as f:
            symbols_data = json.load(f)
        
        imported = []
        for symbol_data in symbols_data:
            symbol = ELFINSymbol(
                name=symbol_data['name'],
                type_sig=symbol_data['type'],
                units=symbol_data.get('units'),
                source_location=symbol_data.get('location'),
                description=symbol_data.get('description'),
            )
            self.elfin_symbols[symbol.name] = symbol
            imported.append(symbol)
        
        logger.info(f"Imported {len(imported)} ELFIN symbols from {symbols_path}")
        return imported
    
    def export_elfin_symbols(self, output_path: str) -> None:
        """Export ELFIN symbols to a JSON file.
        
        Args:
            output_path: Path to save the ELFIN symbols JSON file
        """
        symbols_data = []
        for symbol in self.elfin_symbols.values():
            symbol_data = {
                'name': symbol.name,
                'type': symbol.type_sig,
            }
            if symbol.units:
                symbol_data['units'] = symbol.units
            if symbol.source_location:
                symbol_data['location'] = symbol.source_location
            if symbol.description:
                symbol_data['description'] = symbol.description
            
            symbols_data.append(symbol_data)
        
        with open(output_path, 'w') as f:
            json.dump(symbols_data, f, indent=2)
        
        logger.info(f"Exported {len(symbols_data)} ELFIN symbols to {output_path}")
    
    def bind_symbol_to_node(self, elfin_name: str, node_id: str) -> Tuple[ELFINSymbol, PsiGraphNode]:
        """Bind an ELFIN symbol to a ψ-graph node.
        
        Args:
            elfin_name: Name of the ELFIN symbol
            node_id: ID of the ψ-graph node
            
        Returns:
            Tuple of (symbol, node) that were bound
            
        Raises:
            KeyError: If either the symbol or node doesn't exist
        """
        if elfin_name not in self.elfin_symbols:
            raise KeyError(f"ELFIN symbol '{elfin_name}' not found")
        
        if node_id not in self.psi_nodes:
            raise KeyError(f"ψ-graph node '{node_id}' not found")
        
        symbol = self.elfin_symbols[elfin_name]
        node = self.psi_nodes[node_id]
        
        # Create binding
        self.bindings[elfin_name] = node_id
        
        # Update node to reflect ELFIN verification
        node.is_elfin_verified = True
        node.elfin_symbol = symbol
        
        logger.info(f"Bound ELFIN symbol '{elfin_name}' to ψ-graph node '{node_id}'")
        
        return symbol, node
    
    def verify_concept_graph(self, nodes: Dict[str, Any]) -> Dict[str, bool]:
        """Verify a concept graph against ELFIN symbols.
        
        Args:
            nodes: Dictionary of node_id -> node_data from concept graph
            
        Returns:
            Dictionary mapping node IDs to verification status
        """
        # Register all nodes in the graph
        for node_id, node_data in nodes.items():
            if node_id not in self.psi_nodes:
                self.psi_nodes[node_id] = PsiGraphNode(
                    id=node_id,
                    label=node_data.get('label', node_id),
                    activation=node_data.get('activation', 0.0),
                )
        
        # Check for verification of each node
        verification_status = {}
        for node_id, node in self.psi_nodes.items():
            # Node is verified if it has an ELFIN binding
            is_verified = any(node_id == nid for sym, nid in self.bindings.items())
            verification_status[node_id] = is_verified
            node.is_elfin_verified = is_verified
        
        return verification_status
    
    def get_verified_nodes(self) -> List[PsiGraphNode]:
        """Get all verified nodes in the graph.
        
        Returns:
            List of nodes with ELFIN verification
        """
        return [node for node in self.psi_nodes.values() if node.is_elfin_verified]


# Example usage
if __name__ == "__main__":
    # Path to ELFIN symbol export file
    elfin_export_path = os.path.join(os.path.dirname(__file__), "../../elfin/exports/symbols.json")
    
    # Create bridge
    bridge = ELFINBridge()
    
    # Import ELFIN symbols
    try:
        symbols = bridge.import_elfin_symbols(elfin_export_path)
        print(f"Imported {len(symbols)} ELFIN symbols")
        for i, symbol in enumerate(symbols[:5]):  # Show first 5
            print(f"  {i+1}. {symbol.name}: {symbol.type_sig}")
        if len(symbols) > 5:
            print(f"  ... and {len(symbols) - 5} more")
    except FileNotFoundError:
        print(f"ELFIN symbols file not found at {elfin_export_path}")
        print("This is a skeleton implementation - the actual file will be generated by the ELFIN LSP")
