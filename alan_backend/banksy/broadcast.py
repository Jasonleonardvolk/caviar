"""
Broadcast Module for PCC State

This module provides utilities for sending PCC state updates to the MCP server
for real-time broadcasting to WebSocket clients.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union
import httpx

# Configure logging
logger = logging.getLogger("banksy-broadcast")

# MCP server URL from environment (fallback to localhost)
MCP_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8787/pcc_state")


def emit_pcc(
    step: int,
    phases: np.ndarray,
    spins: np.ndarray,
    energy: float,
    timeout: float = 1.0,
) -> bool:
    """
    Emit PCC state update to the MCP server for WebSocket broadcasting.
    
    Args:
        step: Current simulation step
        phases: Array of phase values [0, 2π)
        spins: Array of spin values (±1)
        energy: System energy
        timeout: HTTP request timeout in seconds
        
    Returns:
        True if successfully sent, False otherwise
    """
    # Create the payload (limit to first 64 elements for efficiency)
    payload = {
        "step": int(step),
        "phases": phases[:64].tolist() if isinstance(phases, np.ndarray) else phases[:64],
        "spins": spins[:64].tolist() if isinstance(spins, np.ndarray) else spins[:64],
        "energy": float(energy)
    }
    
    # Send to MCP server
    try:
        response = httpx.post(
            MCP_URL,
            json=payload,
            timeout=timeout
        )
        
        # Check if request was successful
        if response.status_code == 200:
            logger.debug(f"PCC state broadcast successful: step={step}")
            return True
        else:
            logger.warning(f"PCC state broadcast failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending PCC state to MCP server: {e}")
        return False


def emit_pcc_batch(
    steps: List[int],
    phases_list: List[np.ndarray],
    spins_list: List[np.ndarray],
    energies: List[float],
) -> int:
    """
    Emit multiple PCC states in batch to the MCP server.
    
    Args:
        steps: List of simulation steps
        phases_list: List of phase arrays
        spins_list: List of spin arrays
        energies: List of system energies
        host: MCP server host address
        port: MCP server port
        
    Returns:
        Number of successfully sent updates
    """
    success_count = 0
    
    for step, phases, spins, energy in zip(steps, phases_list, spins_list, energies):
        if emit_pcc(step, phases, spins, energy):
            success_count += 1
    
    return success_count
