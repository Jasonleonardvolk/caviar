"""
Communication Fabric for Dickbox
=================================

Handles inter-service communication via Unix sockets and ZeroMQ.
"""

import asyncio
import json
import logging
import os
import socket
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Union
from abc import ABC, abstractmethod
import struct
import pickle

# Try importing ZeroMQ
try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    zmq = None

# Try importing gRPC for Unix socket support
try:
    import grpc
    from grpc import aio as grpc_aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    grpc = None

logger = logging.getLogger(__name__)


class MessageBus(ABC):
    """Abstract base class for message bus implementations"""
    
    @abstractmethod
    async def publish(self, topic: str, message: Any):
        """Publish a message to a topic"""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic with a callback"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, topic: str):
        """Unsubscribe from a topic"""
        pass
    
    @abstractmethod
    async def start(self):
        """Start the message bus"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the message bus"""
        pass


class ZeroMQBus(MessageBus):
    """
    ZeroMQ-based message bus for pub/sub communication.
    
    Features:
    - Brokerless pub/sub
    - Topic-based routing
    - High performance
    - Multiple transport options (tcp, ipc, inproc)
    - Optional encryption with key rotation
    """
    
    def __init__(self, pub_endpoint: str = "tcp://127.0.0.1:5555",
                 sub_endpoint: str = "tcp://127.0.0.1:5555",
                 enable_encryption: bool = False,
                 keys_dir: Optional[Path] = None):
        if not ZMQ_AVAILABLE:
            raise ImportError("ZeroMQ not available - install pyzmq")
        
        self.pub_endpoint = pub_endpoint
        self.sub_endpoint = sub_endpoint
        self.enable_encryption = enable_encryption
        self.keys_dir = keys_dir or Path("/etc/tori/zmq_keys")
        self.context = None
        self.pub_socket = None
        self.sub_socket = None
        self.subscriptions = {}
        self._running = False
        self._subscriber_task = None
        self._key_reload_task = None
        self.auth = None
    
    async def start(self):
        """Start the ZeroMQ bus"""
        if self._running:
            return
        
        # Create context
        self.context = zmq.asyncio.Context()
        
        # Setup encryption if enabled
        if self.enable_encryption:
            await self._setup_encryption()
        
        # Create publisher socket
        self.pub_socket = self.context.socket(zmq.PUB)
        if self.enable_encryption:
            self.pub_socket.curve_server = True
            self._load_server_keys(self.pub_socket)
        self.pub_socket.bind(self.pub_endpoint)
        
        # Create subscriber socket
        self.sub_socket = self.context.socket(zmq.SUB)
        if self.enable_encryption:
            self._load_client_keys(self.sub_socket)
        self.sub_socket.connect(self.sub_endpoint)
        
        self._running = True
        
        # Start subscriber task
        self._subscriber_task = asyncio.create_task(self._subscriber_loop())
        
        # Start key reload task if encryption enabled
        if self.enable_encryption:
            self._key_reload_task = asyncio.create_task(self._key_reload_loop())
        
        logger.info(f"ZeroMQ bus started - pub: {self.pub_endpoint}, sub: {self.sub_endpoint}, encrypted: {self.enable_encryption}")
    
    async def stop(self):
        """Stop the ZeroMQ bus"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel tasks
        for task in [self._subscriber_task, self._key_reload_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop authenticator
        if self.auth:
            self.auth.stop()
        
        # Close sockets
        if self.pub_socket:
            self.pub_socket.close()
        if self.sub_socket:
            self.sub_socket.close()
        
        # Terminate context
        if self.context:
            self.context.term()
        
        logger.info("ZeroMQ bus stopped")
    
    async def publish(self, topic: str, message: Any):
        """Publish a message to a topic"""
        if not self._running or not self.pub_socket:
            raise RuntimeError("ZeroMQ bus not started")
        
        # Serialize message
        data = pickle.dumps(message)
        
        # Send topic and message
        await self.pub_socket.send_multipart([
            topic.encode('utf-8'),
            data
        ])
        
        logger.debug(f"Published to {topic}: {type(message).__name__}")
    
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic"""
        if not self._running or not self.sub_socket:
            raise RuntimeError("ZeroMQ bus not started")
        
        # Subscribe to topic
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))
        
        # Store callback
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(callback)
        
        logger.info(f"Subscribed to topic: {topic}")
    
    async def unsubscribe(self, topic: str):
        """Unsubscribe from a topic"""
        if not self._running or not self.sub_socket:
            return
        
        # Unsubscribe from topic
        self.sub_socket.setsockopt(zmq.UNSUBSCRIBE, topic.encode('utf-8'))
        
        # Remove callbacks
        if topic in self.subscriptions:
            del self.subscriptions[topic]
        
        logger.info(f"Unsubscribed from topic: {topic}")
    
    async def _subscriber_loop(self):
        """Main subscriber loop"""
        while self._running:
            try:
                # Receive message
                topic_bytes, data = await self.sub_socket.recv_multipart()
                topic = topic_bytes.decode('utf-8')
                
                # Deserialize message
                message = pickle.loads(data)
                
                # Call callbacks
                if topic in self.subscriptions:
                    for callback in self.subscriptions[topic]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(topic, message)
                            else:
                                callback(topic, message)
                        except Exception as e:
                            logger.error(f"Error in callback for {topic}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in subscriber loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _setup_encryption(self):
        """Setup ZeroMQ encryption"""
        if not hasattr(zmq.auth, 'AsyncioAuthenticator'):
            logger.warning("ZeroMQ authentication not available")
            return
        
        # Start authenticator
        self.auth = zmq.auth.AsyncioAuthenticator(self.context)
        self.auth.start()
        
        # Configure curve authentication
        public_keys_dir = self.keys_dir / "public_keys"
        public_keys_dir.mkdir(exist_ok=True)
        
        # Allow all clients with valid certificates
        self.auth.configure_curve(domain='*', location=str(public_keys_dir))
        
        logger.info("ZeroMQ encryption configured")
    
    def _load_server_keys(self, socket):
        """Load server keys for encryption"""
        try:
            # Load current keys
            current_private = self.keys_dir / "current_private.key"
            current_public = self.keys_dir / "current_public.key"
            
            if current_private.exists() and current_public.exists():
                with open(current_private, 'rb') as f:
                    private_key = f.read()
                with open(current_public, 'rb') as f:
                    public_key = f.read()
                
                # Set keys on socket
                socket.curve_secretkey = private_key
                socket.curve_publickey = public_key
                
                logger.info("Loaded server encryption keys")
            else:
                logger.warning("No encryption keys found, generating new ones")
                # Generate keys if not present
                from zmq_key_rotation import ZMQKeyManager
                key_manager = ZMQKeyManager(self.keys_dir)
                key_manager.rotate_keys()
                self._load_server_keys(socket)  # Retry with new keys
                
        except Exception as e:
            logger.error(f"Failed to load server keys: {e}")
    
    def _load_client_keys(self, socket):
        """Load client keys for encryption"""
        try:
            # Load current keys
            current_private = self.keys_dir / "current_private.key"
            current_public = self.keys_dir / "current_public.key"
            
            if current_private.exists() and current_public.exists():
                with open(current_private, 'rb') as f:
                    private_key = f.read()
                with open(current_public, 'rb') as f:
                    public_key = f.read()
                
                # For client, we need server's public key
                server_public = self.keys_dir / "server_public.key"
                if not server_public.exists():
                    # Use same key for now (in production, these would be different)
                    server_public = current_public
                    
                with open(server_public, 'rb') as f:
                    server_key = f.read()
                
                # Set keys on socket
                socket.curve_secretkey = private_key
                socket.curve_publickey = public_key
                socket.curve_serverkey = server_key
                
                logger.info("Loaded client encryption keys")
            else:
                logger.warning("No client encryption keys found")
                
        except Exception as e:
            logger.error(f"Failed to load client keys: {e}")
    
    async def _key_reload_loop(self):
        """Monitor for key rotation events"""
        while self._running:
            try:
                # Check for key rotation event via file monitoring
                # In production, this would use inotify or similar
                await asyncio.sleep(60)  # Check every minute
                
                # Check if keys have changed
                current_private = self.keys_dir / "current_private.key"
                if current_private.exists():
                    stat = current_private.stat()
                    if hasattr(self, '_last_key_mtime'):
                        if stat.st_mtime > self._last_key_mtime:
                            logger.info("Detected key rotation, reloading keys")
                            await self._reload_keys()
                    self._last_key_mtime = stat.st_mtime
                    
            except Exception as e:
                logger.error(f"Error in key reload loop: {e}")
                await asyncio.sleep(60)
    
    async def _reload_keys(self):
        """Reload encryption keys after rotation"""
        # This is simplified - in production, you'd need graceful socket recreation
        logger.info("Reloading ZeroMQ encryption keys")
        
        # Reload keys on sockets
        if self.pub_socket:
            self._load_server_keys(self.pub_socket)
        if self.sub_socket:
            self._load_client_keys(self.sub_socket)
        
        # Notify subscribers of key rotation
        await self.publish("system.key_rotated", {
            "timestamp": datetime.now().isoformat(),
            "service": "zeromq"
        })


class UnixSocketServer:
    """
    Unix domain socket server for local IPC.
    
    Features:
    - Fast local communication
    - File-based permissions
    - Request/response pattern
    """
    
    def __init__(self, socket_path: Path, handler: Callable):
        self.socket_path = socket_path
        self.handler = handler
        self.server = None
        self._running = False
    
    async def start(self):
        """Start the Unix socket server"""
        if self._running:
            return
        
        # Ensure socket directory exists
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing socket
        if self.socket_path.exists():
            self.socket_path.unlink()
        
        # Start server
        self.server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path)
        )
        
        # Set permissions
        os.chmod(self.socket_path, 0o666)
        
        self._running = True
        logger.info(f"Unix socket server started at {self.socket_path}")
    
    async def stop(self):
        """Stop the Unix socket server"""
        if not self._running:
            return
        
        self._running = False
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Remove socket file
        if self.socket_path.exists():
            self.socket_path.unlink()
        
        logger.info("Unix socket server stopped")
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client connection"""
        try:
            while True:
                # Read message length (4 bytes)
                length_bytes = await reader.read(4)
                if not length_bytes:
                    break
                
                length = struct.unpack('!I', length_bytes)[0]
                
                # Read message
                data = await reader.read(length)
                if not data:
                    break
                
                # Deserialize request
                request = json.loads(data.decode('utf-8'))
                
                # Handle request
                response = await self.handler(request)
                
                # Serialize response
                response_data = json.dumps(response).encode('utf-8')
                
                # Send response length and data
                writer.write(struct.pack('!I', len(response_data)))
                writer.write(response_data)
                await writer.drain()
                
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()


class UnixSocketClient:
    """Client for Unix domain socket communication"""
    
    def __init__(self, socket_path: Path):
        self.socket_path = socket_path
        self.reader = None
        self.writer = None
        self._connected = False
    
    async def connect(self):
        """Connect to Unix socket server"""
        if self._connected:
            return
        
        try:
            self.reader, self.writer = await asyncio.open_unix_connection(
                str(self.socket_path)
            )
            self._connected = True
            logger.debug(f"Connected to Unix socket at {self.socket_path}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.socket_path}: {e}")
    
    async def disconnect(self):
        """Disconnect from server"""
        if not self._connected:
            return
        
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        
        self._connected = False
        logger.debug("Disconnected from Unix socket")
    
    async def request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send request and receive response"""
        if not self._connected:
            await self.connect()
        
        try:
            # Serialize request
            request_data = json.dumps(data).encode('utf-8')
            
            # Send request length and data
            self.writer.write(struct.pack('!I', len(request_data)))
            self.writer.write(request_data)
            await self.writer.drain()
            
            # Read response length
            length_bytes = await self.reader.read(4)
            if not length_bytes:
                raise ConnectionError("Connection closed by server")
            
            length = struct.unpack('!I', length_bytes)[0]
            
            # Read response
            response_data = await self.reader.read(length)
            if not response_data:
                raise ConnectionError("Connection closed by server")
            
            # Deserialize response
            return json.loads(response_data.decode('utf-8'))
            
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Request failed: {e}")


class CommunicationFabric:
    """
    Main communication fabric for Dickbox.
    
    Combines:
    - ZeroMQ for pub/sub messaging
    - Unix sockets for RPC
    - Service discovery
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services = {}  # service_name -> socket_path
        self.message_bus = None
        self.unix_servers = {}
        
        # Initialize ZeroMQ if enabled
        if config.get("enable_zeromq", True) and ZMQ_AVAILABLE:
            pub_port = config.get("zeromq_pub_port", 5555)
            self.message_bus = ZeroMQBus(
                pub_endpoint=f"tcp://127.0.0.1:{pub_port}",
                sub_endpoint=f"tcp://127.0.0.1:{pub_port}"
            )
    
    async def start(self):
        """Start the communication fabric"""
        # Start message bus
        if self.message_bus:
            await self.message_bus.start()
        
        logger.info("Communication fabric started")
    
    async def stop(self):
        """Stop the communication fabric"""
        # Stop all Unix servers
        for server in self.unix_servers.values():
            await server.stop()
        
        # Stop message bus
        if self.message_bus:
            await self.message_bus.stop()
        
        logger.info("Communication fabric stopped")
    
    async def register_service(self, service_name: str, socket_path: Path, handler: Callable):
        """
        Register a service with Unix socket endpoint.
        
        Args:
            service_name: Name of the service
            socket_path: Path to Unix socket
            handler: Request handler function
        """
        # Create Unix socket server
        server = UnixSocketServer(socket_path, handler)
        await server.start()
        
        # Track service
        self.services[service_name] = socket_path
        self.unix_servers[service_name] = server
        
        # Announce service via message bus
        if self.message_bus:
            await self.message_bus.publish("service.registered", {
                "service": service_name,
                "socket": str(socket_path),
                "timestamp": asyncio.get_event_loop().time()
            })
        
        logger.info(f"Registered service {service_name} at {socket_path}")
    
    async def unregister_service(self, service_name: str):
        """Unregister a service"""
        if service_name not in self.services:
            return
        
        # Stop Unix server
        if service_name in self.unix_servers:
            await self.unix_servers[service_name].stop()
            del self.unix_servers[service_name]
        
        # Remove from registry
        socket_path = self.services.pop(service_name)
        
        # Announce removal
        if self.message_bus:
            await self.message_bus.publish("service.unregistered", {
                "service": service_name,
                "timestamp": asyncio.get_event_loop().time()
            })
        
        logger.info(f"Unregistered service {service_name}")
    
    async def call_service(self, service_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a service via Unix socket.
        
        Args:
            service_name: Name of the service
            request: Request data
            
        Returns:
            Response from service
        """
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        socket_path = self.services[service_name]
        client = UnixSocketClient(socket_path)
        
        try:
            response = await client.request(request)
            return response
        finally:
            await client.disconnect()
    
    async def publish_event(self, topic: str, event: Any):
        """Publish an event to the message bus"""
        if not self.message_bus:
            logger.warning(f"No message bus available to publish {topic}")
            return
        
        await self.message_bus.publish(topic, event)
    
    async def subscribe_event(self, topic: str, callback: Callable):
        """Subscribe to events on the message bus"""
        if not self.message_bus:
            logger.warning(f"No message bus available to subscribe to {topic}")
            return
        
        await self.message_bus.subscribe(topic, callback)
    
    def get_service_socket(self, service_name: str) -> Optional[Path]:
        """Get Unix socket path for a service"""
        return self.services.get(service_name)
    
    def list_services(self) -> List[str]:
        """List all registered services"""
        return list(self.services.keys())


# Export
__all__ = [
    'CommunicationFabric',
    'ZeroMQBus',
    'UnixSocketServer',
    'UnixSocketClient',
    'MessageBus'
]
