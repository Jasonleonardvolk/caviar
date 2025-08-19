#!/usr/bin/env python3
"""
TorusRegistry - Parquet-based persistence for topological data
Replaces SQLite with mesh-compatible storage
"""

import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import hashlib
import os
from datetime import datetime, timezone

class TorusRegistry:
    """
    Topology-aware registry using Parquet storage
    No database dependencies - pure file-based persistence
    """
    
    SCHEMA = "v1"
    
    def __init__(self, path: Path):
        """
        Initialize registry with configurable path
        
        Args:
            path: Path to Parquet file (auto-created if missing)
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df = self._load()
        self._pending_writes = []
        
    def _load(self) -> pd.DataFrame:
        """Load existing registry or create new one"""
        if self.path.exists():
            try:
                df = pd.read_parquet(self.path)
                rows_loaded = len(df)
                print(f"TorusRegistry loaded {rows_loaded} rows (schema {self.SCHEMA}) from {self.path}")
                return df
            except Exception as e:
                print(f"Warning: Failed to load registry: {e}. Creating new one.")
                
        # Create new DataFrame with schema
        return pd.DataFrame(columns=[
            "shape_id",      # Unique hash of vertices
            "timestamp",     # When recorded
            "betti0",        # 0th Betti number
            "betti1",        # 1st Betti number
            "betti2",        # 2nd Betti number (optional)
            "vertices_hash", # Hash for deduplication
            "coherence_band", # local/global/critical
            "metadata",      # JSON string for extra data
            "_schema"        # Schema version
        ])
    
    def record_shape(self, vertices: np.ndarray, 
                    betti_numbers: Optional[List[float]] = None,
                    coherence_band: str = "local",
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a topological shape with its Betti numbers
        
        Args:
            vertices: Array of vertices defining the shape
            betti_numbers: Pre-computed Betti numbers [b0, b1, b2, ...]
            coherence_band: Current coherence state
            metadata: Additional metadata to store
            
        Returns:
            shape_id of the recorded shape
        """
        # Compute shape hash
        if isinstance(vertices, np.ndarray):
            vertices_bytes = vertices.tobytes()
        else:
            vertices_bytes = np.array(vertices).tobytes()
            
        vertices_hash = hashlib.blake2b(vertices_bytes, digest_size=16).hexdigest()
        shape_id = f"shape_{vertices_hash[:12]}"
        
        # Use provided Betti numbers or compute if needed
        if betti_numbers is None:
            # Import here to avoid circular dependency
            from .topo_ops import betti0_1
            b0, b1 = betti0_1(vertices)
            betti_numbers = [b0, b1]
        
        # Prepare row
        row = {
            "shape_id": shape_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "betti0": float(betti_numbers[0]) if len(betti_numbers) > 0 else 0.0,
            "betti1": float(betti_numbers[1]) if len(betti_numbers) > 1 else 0.0,
            "betti2": float(betti_numbers[2]) if len(betti_numbers) > 2 else None,
            "vertices_hash": vertices_hash,
            "coherence_band": coherence_band,
            "metadata": pd.io.json.dumps(metadata) if metadata else "{}",
            "_schema": self.SCHEMA
        }
        
        # Buffer write
        self._pending_writes.append(row)
        
        # Auto-flush if buffer is large
        if len(self._pending_writes) >= 100:
            self.flush()
            
        return shape_id
    
    def flush(self):
        """Atomically write pending changes to disk"""
        if not self._pending_writes:
            return
            
        # Add pending writes to DataFrame
        new_rows = pd.DataFrame(self._pending_writes)
        self.df = pd.concat([self.df, new_rows], ignore_index=True)
        
        # Atomic write with temp file
        tmp_path = self.path.with_suffix('.tmp')
        try:
            self.df.to_parquet(tmp_path, index=False, compression='snappy')
            tmp_path.replace(self.path)
            self._pending_writes.clear()
        except Exception as e:
            print(f"Error flushing registry: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    
    def query_by_coherence(self, coherence_band: str) -> pd.DataFrame:
        """Query all shapes in a specific coherence band"""
        return self.df[self.df['coherence_band'] == coherence_band].copy()
    
    def query_recent(self, n: int = 100) -> pd.DataFrame:
        """Get n most recent entries"""
        return self.df.tail(n).copy()
    
    def compute_betti_delta(self, shape_id1: str, shape_id2: str) -> Dict[str, float]:
        """Compute Betti number differences between two shapes"""
        row1 = self.df[self.df['shape_id'] == shape_id1]
        row2 = self.df[self.df['shape_id'] == shape_id2]
        
        if row1.empty or row2.empty:
            return {}
            
        return {
            'delta_b0': float(row2.iloc[0]['betti0'] - row1.iloc[0]['betti0']),
            'delta_b1': float(row2.iloc[0]['betti1'] - row1.iloc[0]['betti1']),
            'delta_b2': float(row2.iloc[0]['betti2'] - row1.iloc[0]['betti2']) 
                       if row1.iloc[0]['betti2'] is not None else None
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        if self.df.empty:
            return {
                'total_shapes': 0,
                'coherence_distribution': {},
                'avg_betti': {'b0': 0.0, 'b1': 0.0}
            }
            
        return {
            'total_shapes': len(self.df),
            'pending_writes': len(self._pending_writes),
            'coherence_distribution': self.df['coherence_band'].value_counts().to_dict(),
            'avg_betti': {
                'b0': float(self.df['betti0'].mean()),
                'b1': float(self.df['betti1'].mean()),
                'b2': float(self.df['betti2'].mean()) if self.df['betti2'].notna().any() else None
            },
            'schema_version': self.SCHEMA
        }
    
    def __del__(self):
        """Ensure pending writes are flushed on cleanup"""
        if hasattr(self, '_pending_writes') and self._pending_writes:
            try:
                self.flush()
            except:
                pass  # Best effort on cleanup

# Global registry path configuration
STATE_ROOT = Path(os.getenv("TORI_STATE_ROOT", "/var/lib/tori")).expanduser()
STATE_ROOT.mkdir(parents=True, exist_ok=True)
REG_PATH = STATE_ROOT / "torus_registry.parquet"

# Convenience function
def get_torus_registry(path: Optional[Path] = None) -> TorusRegistry:
    """Get or create TorusRegistry instance"""
    return TorusRegistry(path or REG_PATH)
