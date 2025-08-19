#!/usr/bin/env python3
"""
Python Bridge Server - Handles communication between Node.js and Python modules
"""

import sys
import json
import importlib.util
import importlib
import traceback
import asyncio
import inspect
from typing import Any, Dict, Optional
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, complex):
            return {'__complex__': True, 'real': obj.real, 'imag': obj.imag}
        return super().default(obj)

class PythonBridgeServer:
    def __init__(self, module_path: str):
        self.module_path = module_path
        self.module = None
        self.instance = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def load_module(self):
        """Load the specified Python module"""
        try:
            # Try to load as a module first
            if self.module_path.endswith('.py'):
                spec = importlib.util.spec_from_file_location("bridge_module", self.module_path)
                if spec and spec.loader:
                    self.module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(self.module)
            else:
                # Load as package
                self.module = importlib.import_module(self.module_path)
            
            # Try to instantiate main class or get module directly
            if hasattr(self.module, 'CognitiveEngine'):
                # Initialize with default config
                self.instance = self.module.CognitiveEngine({})
            elif hasattr(self.module, 'UnifiedMemoryVault'):
                self.instance = self.module.UnifiedMemoryVault({})
            elif hasattr(self.module, 'EigenvalueMonitor'):
                self.instance = self.module.EigenvalueMonitor({})
            elif hasattr(self.module, 'LyapunovAnalyzer'):
                self.instance = self.module.LyapunovAnalyzer({})
            elif hasattr(self.module, 'KoopmanOperator'):
                self.instance = self.module.KoopmanOperator({})
            else:
                # No known class, use module directly
                self.instance = self.module
                
            self.send_message({'type': 'ready'})
            
        except Exception as e:
            self.send_message({
                'type': 'error',
                'error': f'Failed to load module: {str(e)}',
                'traceback': traceback.format_exc()
            })
            sys.exit(1)
    
    def send_message(self, message: Dict[str, Any]):
        """Send JSON message to Node.js"""
        print(json.dumps(message, cls=NumpyEncoder))
        sys.stdout.flush()
    
    async def handle_call(self, call_data: Dict[str, Any]):
        """Handle method call from Node.js"""
        call_id = call_data.get('id')
        method_name = call_data.get('method')
        args = call_data.get('args', [])
        
        try:
            # Special built-in methods
            if method_name == 'shutdown':
                self.send_message({
                    'type': 'response',
                    'id': call_id,
                    'result': 'OK'
                })
                await self.shutdown()
                return
            
            elif method_name == 'import_module':
                module_name = args[0] if args else None
                if module_name:
                    importlib.import_module(module_name)
                self.send_message({
                    'type': 'response',
                    'id': call_id,
                    'result': 'OK'
                })
                return
            
            elif method_name == 'get_attr':
                attr_name = args[0] if args else None
                if attr_name and hasattr(self.instance, attr_name):
                    result = getattr(self.instance, attr_name)
                    # Don't try to serialize functions
                    if callable(result):
                        result = f"<function {attr_name}>"
                else:
                    result = None
                self.send_message({
                    'type': 'response',
                    'id': call_id,
                    'result': result
                })
                return
            
            elif method_name == 'set_attr':
                if len(args) >= 2:
                    attr_name, value = args[0], args[1]
                    setattr(self.instance, attr_name, value)
                self.send_message({
                    'type': 'response',
                    'id': call_id,
                    'result': 'OK'
                })
                return
            
            elif method_name == 'eval':
                expression = args[0] if args else ''
                result = eval(expression, {'module': self.module, 'instance': self.instance})
                self.send_message({
                    'type': 'response',
                    'id': call_id,
                    'result': result
                })
                return
            
            elif method_name == 'exec':
                code = args[0] if args else ''
                exec(code, {'module': self.module, 'instance': self.instance})
                self.send_message({
                    'type': 'response',
                    'id': call_id,
                    'result': 'OK'
                })
                return
            
            # Regular method call
            if hasattr(self.instance, method_name):
                method = getattr(self.instance, method_name)
                
                # Check if method is async
                if inspect.iscoroutinefunction(method):
                    result = await method(*args)
                else:
                    result = method(*args)
                
                # Convert numpy arrays and other non-serializable types
                if isinstance(result, np.ndarray):
                    result = result.tolist()
                
                self.send_message({
                    'type': 'response',
                    'id': call_id,
                    'result': result
                })
            else:
                raise AttributeError(f"Method '{method_name}' not found")
            
        except Exception as e:
            # Send error
            self.send_message({
                'type': 'response',
                'id': call_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    def emit_event(self, event_name: str, data: Any):
        """Emit an event to Node.js"""
        self.send_message({
            'type': 'event',
            'event': event_name,
            'data': data
        })
    
    async def shutdown(self):
        """Clean shutdown"""
        # If instance has shutdown method, call it
        if hasattr(self.instance, 'shutdown'):
            if inspect.iscoroutinefunction(self.instance.shutdown):
                await self.instance.shutdown()
            else:
                self.instance.shutdown()
        
        # Stop event loop
        self.loop.stop()
        sys.exit(0)
    
    async def process_input(self):
        """Process incoming messages from stdin"""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        
        await self.loop.connect_read_pipe(lambda: protocol, sys.stdin)
        
        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                    
                line = line.decode().strip()
                if not line:
                    continue
                
                try:
                    call_data = json.loads(line)
                    await self.handle_call(call_data)
                except json.JSONDecodeError:
                    self.send_message({
                        'type': 'error',
                        'error': f'Invalid JSON: {line}'
                    })
            except Exception as e:
                self.send_message({
                    'type': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
    
    def run(self):
        """Main event loop"""
        self.load_module()
        
        try:
            self.loop.run_until_complete(self.process_input())
        except KeyboardInterrupt:
            self.loop.run_until_complete(self.shutdown())
        finally:
            self.loop.close()


def decode_complex(dct):
    """Decode complex numbers from JSON"""
    if '__complex__' in dct:
        return complex(dct['real'], dct['imag'])
    return dct


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python_bridge_server.py <module_path>")
        sys.exit(1)
    
    # Set up JSON decoding for complex numbers
    json.loads = lambda s: json.loads(s, object_hook=decode_complex)
    
    server = PythonBridgeServer(sys.argv[1])
    server.run()
