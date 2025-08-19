"""
Interactions module for ELFIN stability framework.

This module provides a lightweight, serializable data model for tracking
all interactions with stability verification components.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import json
import pathlib

@dataclass
class Interaction:
    """
    Represents a single interaction with a stability verification component.
    
    Each interaction captures an action (like a verification attempt), its context,
    and eventually its result. Interactions form an append-only log that can be
    used for audit, debugging, and machine learning.
    
    Attributes:
        timestamp: ISO-8601 UTC timestamp of when the interaction occurred
        action: Type of action (e.g., "verify", "counterexample", "param_tune")
        meta: Arbitrary context for the interaction
        result: Optional result of the action
    """
    timestamp: str
    action: str
    meta: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    
    @staticmethod
    def now(action: str, **meta) -> "Interaction":
        """
        Create a new interaction with the current timestamp.
        
        Args:
            action: Type of action being performed
            **meta: Arbitrary keyword arguments for context
            
        Returns:
            New Interaction instance with current timestamp
        """
        return Interaction(
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            meta=meta
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert interaction to a dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert interaction to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Interaction":
        """Create an interaction from a dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Interaction":
        """Create an interaction from a JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_reference(self) -> str:
        """
        Get a unique reference to this interaction.
        
        Returns:
            String in the format "timestamp#action"
        """
        return f"{self.timestamp}#{self.action}"


@dataclass
class InteractionLog:
    """
    An append-only log of interactions.
    
    This class manages a collection of interactions and handles persistence
    to disk in JSONL format.
    
    Attributes:
        interactions: List of interactions in chronological order
    """
    interactions: List[Interaction] = field(default_factory=list)
    
    def append(self, interaction: Interaction) -> None:
        """
        Add an interaction to the log.
        
        Args:
            interaction: Interaction to add
        """
        self.interactions.append(interaction)
    
    def save(self, path: Union[str, pathlib.Path]) -> None:
        """
        Save the interaction log to a JSONL file.
        
        Args:
            path: Path to save the log to
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("w", encoding="utf-8") as f:
            for interaction in self.interactions:
                f.write(interaction.to_json() + "\n")
    
    def append_and_persist(self, interaction: Interaction, path: Union[str, pathlib.Path]) -> None:
        """
        Add an interaction to the log and append it to a JSONL file.
        
        Args:
            interaction: Interaction to add
            path: Path to append the interaction to
        """
        self.append(interaction)
        
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("a", encoding="utf-8") as f:
            f.write(interaction.to_json() + "\n")
    
    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "InteractionLog":
        """
        Load an interaction log from a JSONL file.
        
        Args:
            path: Path to load the log from
            
        Returns:
            InteractionLog with loaded interactions
        """
        path = pathlib.Path(path)
        
        if not path.exists():
            return cls()
        
        log = cls()
        
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    log.append(Interaction.from_json(line))
        
        return log
    
    def get_by_reference(self, reference: str) -> Optional[Interaction]:
        """
        Get an interaction by its reference.
        
        Args:
            reference: Interaction reference in the format "timestamp#action"
            
        Returns:
            Matching interaction or None if not found
        """
        # Split reference into timestamp and action
        try:
            timestamp, action = reference.split("#", 1)
        except ValueError:
            return None
        
        # Find matching interaction
        for interaction in self.interactions:
            if interaction.timestamp == timestamp and interaction.action == action:
                return interaction
        
        return None
    
    def filter(self, action: Optional[str] = None, **meta_filters) -> "InteractionLog":
        """
        Filter interactions by action and metadata.
        
        Args:
            action: Optional action to filter by
            **meta_filters: Metadata key-value pairs to filter by
            
        Returns:
            New InteractionLog containing only matching interactions
        """
        filtered = InteractionLog()
        
        for interaction in self.interactions:
            # Check action if specified
            if action is not None and interaction.action != action:
                continue
            
            # Check metadata filters
            match = True
            for key, value in meta_filters.items():
                if key not in interaction.meta or interaction.meta[key] != value:
                    match = False
                    break
            
            if match:
                filtered.append(interaction)
        
        return filtered
    
    def tail(self, n: int) -> "InteractionLog":
        """
        Get the last n interactions.
        
        Args:
            n: Number of interactions to return
            
        Returns:
            New InteractionLog containing only the last n interactions
        """
        return InteractionLog(self.interactions[-n:])
