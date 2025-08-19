"""
AST Visitor classes for the ELFIN language.

This module defines the Visitor and NodeTransformer base classes for traversing
and transforming AST nodes.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic, Dict, List, Callable, Optional, Type

from .nodes import Node


T = TypeVar('T')
N = TypeVar('N', bound=Node)


class Visitor(Generic[T], ABC):
    """Base class for AST visitors."""
    
    def visit(self, node: Node) -> T:
        """Visit a node and return a result."""
        # Get the method name from the node class
        method_name = f"visit_{node.__class__.__name__}"
        
        # Get the method from this class
        method = getattr(self, method_name, self.generic_visit)
        
        # Call the method with the node
        return method(node)
    
    def generic_visit(self, node: Node) -> T:
        """Default visitor method."""
        # By default, just return a default value
        return self.get_default_value()
    
    @abstractmethod
    def get_default_value(self) -> T:
        """Get the default value to return from a visitor method."""
        pass


class NodeVisitor(Visitor[None]):
    """Visitor that doesn't return a value."""
    
    def get_default_value(self) -> None:
        """Get the default value (None)."""
        return None


class NodeCollector(Visitor[List[N]]):
    """Visitor that collects nodes of a specific type."""
    
    def __init__(self, node_type: Type[N]):
        self.node_type = node_type
        self.nodes: List[N] = []
    
    def get_default_value(self) -> List[N]:
        """Get the default value (empty list)."""
        return []
    
    def generic_visit(self, node: Node) -> List[N]:
        """Visit a node and collect it if it's the right type."""
        if isinstance(node, self.node_type):
            self.nodes.append(node)
        
        # Visit all children
        for key, value in vars(node).items():
            if isinstance(value, Node):
                self.visit(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Node):
                        self.visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, Node):
                        self.visit(item)
        
        return self.nodes


class NodeTransformer(Visitor[Node]):
    """Base class for AST transformers."""
    
    def get_default_value(self) -> Node:
        """Get the default value (the original node)."""
        # This will never be called because generic_visit is overridden
        raise NotImplementedError
    
    def generic_visit(self, node: Node) -> Node:
        """Default transformer method."""
        # By default, transform all children
        changes = False
        
        # Transform all children in the node's attributes
        for key, value in vars(node).items():
            if isinstance(value, Node):
                new_value = self.visit(value)
                if new_value is not value:
                    setattr(node, key, new_value)
                    changes = True
            elif isinstance(value, list):
                new_list = []
                list_changed = False
                
                for i, item in enumerate(value):
                    if isinstance(item, Node):
                        new_item = self.visit(item)
                        if new_item is not item:
                            list_changed = True
                            if new_item is not None:
                                new_list.append(new_item)
                        else:
                            new_list.append(item)
                    else:
                        new_list.append(item)
                
                if list_changed:
                    setattr(node, key, new_list)
                    changes = True
            elif isinstance(value, dict):
                new_dict = {}
                dict_changed = False
                
                for k, item in value.items():
                    if isinstance(item, Node):
                        new_item = self.visit(item)
                        if new_item is not item:
                            dict_changed = True
                            if new_item is not None:
                                new_dict[k] = new_item
                        else:
                            new_dict[k] = item
                    else:
                        new_dict[k] = item
                
                if dict_changed:
                    setattr(node, key, new_dict)
                    changes = True
        
        # Return the transformed node
        return node
