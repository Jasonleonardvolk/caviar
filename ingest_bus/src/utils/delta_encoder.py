"""
Delta Protocol

Efficient delta-based transport protocol for high-frequency updates
with sequence tracking and re-sync capabilities.

Ported from TypeScript to Python for ingest-bus service.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple, TypeVar, Callable, Generic, Union

# Type variable for generic state
T = TypeVar('T')

# Constants
DEFAULT_MAX_HISTORY = 10
MAX_SEQUENCE = 65535  # 16-bit counter


class DeltaEncoder(Generic[T]):
    """
    Delta encoder class
    
    Encodes state updates as efficient deltas with sequence tracking and
    re-sync capabilities.
    """
    
    def __init__(self, 
                 max_history: int = DEFAULT_MAX_HISTORY,
                 require_ack: bool = False,
                 ack_timeout: int = 5000,
                 backoff_factor: float = 1.5,
                 max_backoff: int = 30000,
                 on_metrics: Optional[Callable[[Dict[str, float]], None]] = None):
        """
        Create a new delta encoder
        
        Args:
            max_history: Maximum previous states to keep
            require_ack: Whether to require ACKs by default
            ack_timeout: Timeout in ms before considering packet lost
            backoff_factor: Exponential backoff factor for resync requests
            max_backoff: Maximum backoff time in ms
            on_metrics: Callback for metrics reporting
        """
        self.sequence = 0
        self.history: List[T] = []
        self.options = {
            'max_history': max_history,
            'require_ack': require_ack,
            'ack_timeout': ack_timeout,
            'backoff_factor': backoff_factor,
            'max_backoff': max_backoff,
            'on_metrics': on_metrics or (lambda _: None)
        }
        self.pending_acks: Dict[int, Dict[str, Any]] = {}
        self.resync_requested = False
        self.last_resync_time = 0
    
    def encode(self, state: T, require_ack: Optional[bool] = None) -> Dict[str, Any]:
        """
        Encode a state update
        
        Args:
            state: Current full state
            require_ack: Whether to require an ACK for this packet
            
        Returns:
            Delta packet
        """
        need_ack = require_ack if require_ack is not None else self.options['require_ack']
        
        # Increment sequence number with rollover
        self.sequence = (self.sequence + 1) % MAX_SEQUENCE
        
        # If resync requested or first packet, send full state
        if self.resync_requested or not self.history:
            packet = {
                'sequence': self.sequence,
                'baseState': state,
                'requireAck': need_ack,
                'timestamp': int(time.time() * 1000)
            }
            
            # Update history
            self.history = [state]
            self.resync_requested = False
            
            # Track pending ACK if required
            if need_ack:
                now = int(time.time() * 1000)
                self.pending_acks[self.sequence] = {
                    'timestamp': now,
                    'state': state,
                    'attempts': 0,
                    'nextRetry': now + self.options['ack_timeout']
                }
            
            return packet
        
        # Calculate delta from previous state
        previous_state = self.history[-1]
        deltas = self._calculate_deltas(previous_state, state)
        
        # If deltas are larger than full state, send full state instead
        if len(json.dumps(deltas)) > len(json.dumps(state)) * 0.7:
            packet = {
                'sequence': self.sequence,
                'baseState': state,
                'requireAck': need_ack,
                'timestamp': int(time.time() * 1000)
            }
            
            # Calculate size metrics
            full_state_size = len(json.dumps(state))
            deltas_size = len(json.dumps(deltas))
            ratio = deltas_size / full_state_size
            
            # If metrics collection is provided, report sizes
            if self.options['on_metrics']:
                self.options['on_metrics']({
                    'deltaFullRatio': ratio,
                    'fullStateSize': full_state_size,
                    'deltaSize': deltas_size
                })
            
            # Update history
            self._add_to_history(state)
            
            # Track pending ACK if required
            if need_ack:
                now = int(time.time() * 1000)
                self.pending_acks[self.sequence] = {
                    'timestamp': now,
                    'state': state,
                    'attempts': 0,
                    'nextRetry': now + self.options['ack_timeout']
                }
            
            return packet
        
        # Create delta packet
        packet = {
            'sequence': self.sequence,
            'deltas': deltas,
            'requireAck': need_ack,
            'timestamp': int(time.time() * 1000)
        }
        
        # Update history
        self._add_to_history(state)
        
        # Track pending ACK if required
        if need_ack:
            now = int(time.time() * 1000)
            self.pending_acks[self.sequence] = {
                'timestamp': now,
                'state': state,
                'attempts': 0,
                'nextRetry': now + self.options['ack_timeout']
            }
        
        return packet
    
    def handle_ack(self, ack: Dict[str, Any]) -> bool:
        """
        Handle an acknowledgment
        
        Args:
            ack: The acknowledgment packet
            
        Returns:
            Whether the ACK was handled successfully
        """
        # Clear pending ACK
        if ack['sequence'] in self.pending_acks:
            del self.pending_acks[ack['sequence']]
            
            # Handle resync request
            if ack['status'] == 'resync':
                self.resync_requested = True
            
            return True
        
        return False
    
    def check_ack_timeouts(self) -> List[Dict[str, Any]]:
        """
        Check for ACK timeouts and return timed-out states
        Implements exponential backoff for retries
        
        Returns:
            Array of timed-out states for retry
        """
        now = int(time.time() * 1000)
        timed_out = []
        
        # Check if we can request a resync (with backoff)
        can_request_resync = (now - self.last_resync_time) > min(
            self.options['ack_timeout'] * 2, self.options['max_backoff']
        )
        
        sequences_to_check = list(self.pending_acks.keys())
        
        for sequence in sequences_to_check:
            entry = self.pending_acks[sequence]
            
            # Only check entries ready for retry
            if now >= entry['nextRetry']:
                # Increment attempt counter
                entry['attempts'] += 1
                
                # For extreme cases, eventually give up
                if entry['attempts'] > 10:
                    del self.pending_acks[sequence]
                    continue
                
                # Add to timed-out list
                timed_out.append({'sequence': sequence, 'state': entry['state']})
                
                # Calculate exponential backoff for next retry
                backoff_time = min(
                    self.options['ack_timeout'] * (self.options['backoff_factor'] ** entry['attempts']),
                    self.options['max_backoff']
                )
                
                # Update next retry time
                entry['nextRetry'] = now + backoff_time
                
                # If many packets are timing out, request a full resync (with backoff)
                if len(timed_out) > 3 and can_request_resync:
                    self.resync_requested = True
                    self.last_resync_time = now
                    # Clear pending ACKs if we're going to resync anyway
                    self.pending_acks.clear()
                    break  # Stop checking others, we'll resync
        
        return timed_out
    
    def _add_to_history(self, state: T) -> None:
        """
        Add a state to the history
        
        Args:
            state: The state to add
        """
        self.history.append(state)
        
        # Trim history if needed
        if len(self.history) > self.options['max_history']:
            self.history.pop(0)
    
    def _calculate_deltas(self, old_state: T, new_state: T) -> List[Dict[str, Any]]:
        """
        Calculate deltas between two states
        
        This is a simple implementation. More sophisticated diff algorithms
        could be used for specific state types.
        
        Args:
            old_state: Previous state
            new_state: Current state
            
        Returns:
            Array of changes
        """
        # For objects, calculate property-level deltas
        if isinstance(old_state, dict) and isinstance(new_state, dict):
            deltas = []
            
            # For objects, track changed properties
            old_keys = list(old_state.keys())
            new_keys = list(new_state.keys())
            
            # Handle added and modified properties
            for key in new_keys:
                old_value = old_state.get(key)
                new_value = new_state.get(key)
                
                if old_value is None and key not in old_state:
                    # New property
                    deltas.append({'op': 'add', 'path': [key], 'value': new_value})
                elif json.dumps(old_value) != json.dumps(new_value):
                    # Modified property
                    deltas.append({'op': 'replace', 'path': [key], 'value': new_value})
            
            # Handle removed properties
            for key in old_keys:
                if key not in new_keys:
                    deltas.append({'op': 'remove', 'path': [key]})
            
            return deltas
        
        # For arrays, track modified indices
        elif isinstance(old_state, list) and isinstance(new_state, list):
            deltas = []
            
            for i, value in enumerate(new_state):
                if i >= len(old_state):
                    # New element
                    deltas.append({'op': 'add', 'path': [i], 'value': value})
                elif json.dumps(old_state[i]) != json.dumps(value):
                    # Modified element
                    deltas.append({'op': 'replace', 'path': [i], 'value': value})
            
            # Handle removed elements
            if len(old_state) > len(new_state):
                deltas.append({'op': 'truncate', 'path': [], 'length': len(new_state)})
            
            return deltas
        
        # For primitive types, return the new value if different
        if old_state != new_state:
            return [{'op': 'replace', 'path': [], 'value': new_state}]
        
        return []


class DeltaDecoder(Generic[T]):
    """
    Delta decoder class
    
    Decodes delta packets into full states.
    """
    
    def __init__(self, on_resync_needed: Optional[Callable[[], None]] = None):
        """
        Create a new delta decoder
        
        Args:
            on_resync_needed: Callback when resync is needed
        """
        self.current_state: Optional[T] = None
        self.last_sequence = -1
        self.on_resync_needed = on_resync_needed or (lambda: None)
    
    def decode(self, packet: Dict[str, Any]) -> Tuple[Optional[T], Optional[Dict[str, Any]]]:
        """
        Decode a delta packet
        
        Args:
            packet: The delta packet
            
        Returns:
            The decoded state and ACK packet
        """
        # Check for out-of-order packets
        if self.last_sequence != -1:
            expected_sequence = (self.last_sequence + 1) % MAX_SEQUENCE
            
            # Handle sequence mismatch
            if packet['sequence'] != expected_sequence:
                # Skip older packets
                if ((packet['sequence'] < expected_sequence and expected_sequence - packet['sequence'] < 1000) or
                    (packet['sequence'] > expected_sequence and packet['sequence'] - expected_sequence > 1000)):
                    # Old packet, ignore
                    return self.current_state, None
                
                # Otherwise, request resync
                self.on_resync_needed()
                
                if packet.get('requireAck'):
                    return None, {
                        'sequence': packet['sequence'],
                        'status': 'resync',
                        'timestamp': int(time.time() * 1000)
                    }
                
                return None, None
        
        # Update sequence
        self.last_sequence = packet['sequence']
        
        # Handle base state
        if 'baseState' in packet:
            self.current_state = packet['baseState']
            
            if packet.get('requireAck'):
                return self.current_state, {
                    'sequence': packet['sequence'],
                    'status': 'ok',
                    'timestamp': int(time.time() * 1000)
                }
            
            return self.current_state, None
        
        # Handle deltas
        if 'deltas' in packet and self.current_state is not None:
            try:
                self._apply_deltas(packet['deltas'])
                
                if packet.get('requireAck'):
                    return self.current_state, {
                        'sequence': packet['sequence'],
                        'status': 'ok',
                        'timestamp': int(time.time() * 1000)
                    }
                
                return self.current_state, None
            except Exception:
                # Failed to apply deltas, request resync
                self.on_resync_needed()
                
                if packet.get('requireAck'):
                    return self.current_state, {
                        'sequence': packet['sequence'],
                        'status': 'resync',
                        'timestamp': int(time.time() * 1000)
                    }
                
                return self.current_state, None
        
        # Missing required data
        if packet.get('requireAck'):
            return self.current_state, {
                'sequence': packet['sequence'],
                'status': 'resync',
                'timestamp': int(time.time() * 1000)
            }
        
        return self.current_state, None
    
    def _apply_deltas(self, deltas: List[Dict[str, Any]]) -> None:
        """
        Apply deltas to the current state
        
        Args:
            deltas: Array of deltas to apply
        """
        if self.current_state is None:
            raise ValueError('Cannot apply deltas without a base state')
        
        for delta in deltas:
            self._apply_delta(delta)
    
    def _apply_delta(self, delta: Dict[str, Any]) -> None:
        """
        Apply a single delta to the current state
        
        Args:
            delta: The delta to apply
        """
        if self.current_state is None:
            return
        
        op = delta['op']
        path = delta['path']
        
        if not path:
            # Root-level operations
            if op == 'replace':
                self.current_state = delta['value']
            elif op == 'truncate' and isinstance(self.current_state, list) and 'length' in delta:
                self.current_state = self.current_state[:delta['length']]
            return
        
        # Traverse the path to find the target object
        target = self.current_state
        last_key = path[-1]
        
        for i in range(len(path) - 1):
            key = path[i]
            if isinstance(target, dict):
                if key not in target:
                    target[key] = [] if isinstance(path[i + 1], int) else {}
            elif isinstance(target, list):
                if key >= len(target):
                    target.extend([None] * (key - len(target) + 1))
            else:
                raise ValueError(f"Cannot traverse path in state of type {type(target)}")
            target = target[key]
        
        # Apply the operation
        if op == 'add' or op == 'replace':
            target[last_key] = delta['value']
        elif op == 'remove':
            if isinstance(target, list) and isinstance(last_key, int):
                if 0 <= last_key < len(target):
                    target.pop(last_key)
            elif isinstance(target, dict) and last_key in target:
                del target[last_key]
