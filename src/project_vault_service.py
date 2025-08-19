"""
Project Vault Service

This module provides secure storage for sensitive information like API keys,
passwords, and tokens that are detected during code import. It implements:

- AES-GCM encryption for secure storage
- Multiple backend storage options (file & OS keychain)
- REST API for vault operations
"""

import os
import json
import time
import base64
import hashlib
import hmac
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import secrets

# For encryption
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# For keychain integration (OS-specific backends)
try:
    import keyring
    KEYCHAIN_AVAILABLE = True
except ImportError:
    KEYCHAIN_AVAILABLE = False

# For API server
from flask import Flask, request, jsonify


class VaultEncryption:
    """Handles encryption and decryption of vault data using AES-GCM."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize encryption with optional master key.
        
        Args:
            master_key: 32-byte master key. If None, a new one is generated.
        """
        self.master_key = master_key or secrets.token_bytes(32)
    
    def derive_key(self, salt: bytes, info: bytes = b'') -> bytes:
        """
        Derive a key using PBKDF2.
        
        Args:
            salt: Salt for key derivation
            info: Optional context info
            
        Returns:
            Derived key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(self.master_key + info)
    
    def encrypt(self, plaintext: Union[str, bytes], context: str = '') -> Dict[str, str]:
        """
        Encrypt data using AES-GCM.
        
        Args:
            plaintext: Data to encrypt
            context: Additional context (e.g. secret name)
            
        Returns:
            Dictionary with encrypted data, nonce, and salt
        """
        # Convert plaintext to bytes if it's a string
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        # Generate salt and nonce
        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(12)
        
        # Derive key with salt
        context_bytes = context.encode('utf-8') if context else b''
        key = self.derive_key(salt, context_bytes)
        
        # Encrypt with AES-GCM
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, context_bytes)
        
        # Return as base64 for storage
        return {
            'ciphertext': base64.b64encode(ciphertext).decode('ascii'),
            'nonce': base64.b64encode(nonce).decode('ascii'),
            'salt': base64.b64encode(salt).decode('ascii'),
        }
    
    def decrypt(self, encrypted_data: Dict[str, str], context: str = '') -> bytes:
        """
        Decrypt data using AES-GCM.
        
        Args:
            encrypted_data: Dictionary with ciphertext, nonce, and salt
            context: Additional context (must match encryption context)
            
        Returns:
            Decrypted data as bytes
        """
        # Decode base64 components
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        nonce = base64.b64decode(encrypted_data['nonce'])
        salt = base64.b64decode(encrypted_data['salt'])
        
        # Derive key with salt
        context_bytes = context.encode('utf-8') if context else b''
        key = self.derive_key(salt, context_bytes)
        
        # Decrypt with AES-GCM
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, context_bytes)


class VaultBackend:
    """Base class for vault storage backends."""
    
    def put(self, key: str, value: Dict[str, str]) -> None:
        """
        Store an encrypted value.
        
        Args:
            key: Secret name/identifier
            value: Encrypted data dictionary
        """
        raise NotImplementedError("Backend must implement put()")
    
    def get(self, key: str) -> Optional[Dict[str, str]]:
        """
        Retrieve an encrypted value.
        
        Args:
            key: Secret name/identifier
            
        Returns:
            Encrypted data dictionary or None if not found
        """
        raise NotImplementedError("Backend must implement get()")
    
    def delete(self, key: str) -> bool:
        """
        Delete a value.
        
        Args:
            key: Secret name/identifier
            
        Returns:
            True if deleted, False if not found
        """
        raise NotImplementedError("Backend must implement delete()")
    
    def list_keys(self) -> List[str]:
        """
        List all stored keys.
        
        Returns:
            List of key names
        """
        raise NotImplementedError("Backend must implement list_keys()")


class FileBackend(VaultBackend):
    """Stores vault data in an encrypted file."""
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize file backend.
        
        Args:
            file_path: Path to vault file
        """
        self.file_path = Path(file_path)
        self.data = {}
        self._load()
    
    def _load(self) -> None:
        """Load data from file."""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading vault file: {e}")
                self.data = {}
    
    def _save(self) -> None:
        """Save data to file."""
        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.data, f, indent=2)
        except IOError as e:
            print(f"Error saving vault file: {e}")
    
    def put(self, key: str, value: Dict[str, str]) -> None:
        """Store an encrypted value."""
        self.data[key] = value
        self._save()
    
    def get(self, key: str) -> Optional[Dict[str, str]]:
        """Retrieve an encrypted value."""
        return self.data.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete a value."""
        if key in self.data:
            del self.data[key]
            self._save()
            return True
        return False
    
    def list_keys(self) -> List[str]:
        """List all stored keys."""
        return list(self.data.keys())


class KeychainBackend(VaultBackend):
    """Stores vault data in the OS keychain."""
    
    def __init__(self, service_name: str = "ALAN_IDE_Vault"):
        """
        Initialize keychain backend.
        
        Args:
            service_name: Service name for keychain entries
        """
        self.service_name = service_name
        
        if not KEYCHAIN_AVAILABLE:
            raise ImportError("keyring package not available, cannot use KeychainBackend")
        
        # Keys are stored separately in a file
        self.keys_file = Path(os.path.expanduser("~/.alan_ide/vault_keys.json"))
        self._keys = set()
        self._load_keys()
    
    def _load_keys(self) -> None:
        """Load key names from keys file."""
        if self.keys_file.exists():
            try:
                with open(self.keys_file, 'r') as f:
                    self._keys = set(json.load(f))
            except (json.JSONDecodeError, IOError):
                self._keys = set()
    
    def _save_keys(self) -> None:
        """Save key names to keys file."""
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.keys_file, 'w') as f:
                json.dump(list(self._keys), f)
        except IOError as e:
            print(f"Error saving vault keys: {e}")
    
    def put(self, key: str, value: Dict[str, str]) -> None:
        """Store an encrypted value."""
        keyring.set_password(self.service_name, key, json.dumps(value))
        self._keys.add(key)
        self._save_keys()
    
    def get(self, key: str) -> Optional[Dict[str, str]]:
        """Retrieve an encrypted value."""
        try:
            data = keyring.get_password(self.service_name, key)
            return json.loads(data) if data else None
        except Exception:
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a value."""
        if key in self._keys:
            try:
                keyring.delete_password(self.service_name, key)
                self._keys.remove(key)
                self._save_keys()
                return True
            except Exception:
                return False
        return False
    
    def list_keys(self) -> List[str]:
        """List all stored keys."""
        return list(self._keys)


class VaultService:
    """
    Project Vault Service with multiple backend support.
    
    Provides secure storage for sensitive information like API keys,
    passwords, and tokens. Supports multiple storage backends with
    fallback capability.
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize vault service.
        
        Args:
            master_key: Optional master key for encryption
        """
        # Initialize encryption
        self.encryption = VaultEncryption(master_key)
        
        # Initialize backends
        self.backends = []
        
        # Default file backend
        vault_path = os.environ.get(
            "ALAN_VAULT_PATH", 
            os.path.expanduser("~/.alan_ide/vault.json")
        )
        self.add_backend(FileBackend(vault_path))
        
        # Keychain backend if available
        if KEYCHAIN_AVAILABLE:
            try:
                self.add_backend(KeychainBackend())
            except ImportError:
                pass
    
    def add_backend(self, backend: VaultBackend) -> None:
        """
        Add a storage backend.
        
        Args:
            backend: Backend instance
        """
        self.backends.append(backend)
    
    def put(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a secret securely.
        
        Args:
            key: Secret name (e.g. 'aws_access_key')
            value: Secret value
            metadata: Optional metadata (e.g. source file, description)
            
        Returns:
            True if stored successfully
        """
        # Encrypt the value
        encrypted = self.encryption.encrypt(value, key)
        
        # Add metadata
        if metadata:
            encrypted['metadata'] = metadata
        
        # Add timestamp
        encrypted['timestamp'] = time.time()
        
        # Store in all backends
        success = False
        for backend in self.backends:
            try:
                backend.put(key, encrypted)
                success = True
            except Exception as e:
                print(f"Error storing in backend {backend.__class__.__name__}: {e}")
                continue
        
        return success
    
    def get(self, key: str) -> Optional[str]:
        """
        Retrieve a secret.
        
        Args:
            key: Secret name
            
        Returns:
            Decrypted secret value or None if not found
        """
        # Try each backend until we find the key
        for backend in self.backends:
            try:
                encrypted = backend.get(key)
                if encrypted:
                    # Remove metadata before decrypting
                    encrypt_data = {k: v for k, v in encrypted.items() 
                                    if k not in ('metadata', 'timestamp')}
                    
                    # Decrypt and return
                    plaintext = self.encryption.decrypt(encrypt_data, key)
                    return plaintext.decode('utf-8')
            except Exception as e:
                print(f"Error retrieving from backend {backend.__class__.__name__}: {e}")
                continue
        
        return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a secret.
        
        Args:
            key: Secret name
            
        Returns:
            True if deleted from any backend
        """
        # Delete from all backends
        success = False
        for backend in self.backends:
            try:
                if backend.delete(key):
                    success = True
            except Exception as e:
                print(f"Error deleting from backend {backend.__class__.__name__}: {e}")
                continue
        
        return success
    
    def list_keys(self) -> List[str]:
        """
        List all secret names.
        
        Returns:
            List of secret names
        """
        all_keys = set()
        
        # Collect keys from all backends
        for backend in self.backends:
            try:
                keys = backend.list_keys()
                all_keys.update(keys)
            except Exception as e:
                print(f"Error listing keys from backend {backend.__class__.__name__}: {e}")
                continue
        
        return list(all_keys)
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a secret.
        
        Args:
            key: Secret name
            
        Returns:
            Metadata dictionary or None if not found
        """
        # Try each backend until we find the key
        for backend in self.backends:
            try:
                encrypted = backend.get(key)
                if encrypted and 'metadata' in encrypted:
                    return encrypted['metadata']
            except Exception as e:
                print(f"Error retrieving metadata from backend {backend.__class__.__name__}: {e}")
                continue
        
        return None


# REST API for vault
app = Flask(__name__)
vault_service = None  # Will be initialized in main()

@app.route('/vault/put', methods=['POST'])
def api_put():
    """API endpoint to store a secret."""
    data = request.json
    
    if not data or 'key' not in data or 'value' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    key = data['key']
    value = data['value']
    metadata = data.get('metadata', {})
    
    success = vault_service.put(key, value, metadata)
    
    if success:
        return jsonify({'status': 'success', 'message': f'Secret {key} stored successfully'})
    else:
        return jsonify({'error': 'Failed to store secret'}), 500

@app.route('/vault/get', methods=['POST'])
def api_get():
    """API endpoint to retrieve a secret."""
    data = request.json
    
    if not data or 'key' not in data:
        return jsonify({'error': 'Missing key field'}), 400
    
    key = data['key']
    value = vault_service.get(key)
    
    if value is not None:
        return jsonify({'status': 'success', 'value': value})
    else:
        return jsonify({'error': 'Secret not found'}), 404

@app.route('/vault/delete', methods=['POST'])
def api_delete():
    """API endpoint to delete a secret."""
    data = request.json
    
    if not data or 'key' not in data:
        return jsonify({'error': 'Missing key field'}), 400
    
    key = data['key']
    success = vault_service.delete(key)
    
    if success:
        return jsonify({'status': 'success', 'message': f'Secret {key} deleted successfully'})
    else:
        return jsonify({'error': 'Secret not found or could not be deleted'}), 404

@app.route('/vault/list', methods=['GET'])
def api_list():
    """API endpoint to list all secret names."""
    keys = vault_service.list_keys()
    return jsonify({'status': 'success', 'keys': keys})

@app.route('/vault/metadata', methods=['POST'])
def api_metadata():
    """API endpoint to get metadata for a secret."""
    data = request.json
    
    if not data or 'key' not in data:
        return jsonify({'error': 'Missing key field'}), 400
    
    key = data['key']
    metadata = vault_service.get_metadata(key)
    
    if metadata is not None:
        return jsonify({'status': 'success', 'metadata': metadata})
    else:
        return jsonify({'error': 'Secret not found or no metadata available'}), 404


def create_cli_app() -> VaultService:
    """Create a vault service for CLI usage."""
    return VaultService()


def start_api_server(host='127.0.0.1', port=5000) -> None:
    """Start the API server."""
    global vault_service
    vault_service = VaultService()
    app.run(host=host, port=port)


def main():
    """Command-line interface for the vault service."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Project Vault Service')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Put command
    put_parser = subparsers.add_parser('put', help='Store a secret')
    put_parser.add_argument('key', help='Secret name')
    put_parser.add_argument('value', help='Secret value')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Retrieve a secret')
    get_parser.add_argument('key', help='Secret name')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a secret')
    delete_parser.add_argument('key', help='Secret name')
    
    # List command
    subparsers.add_parser('list', help='List all secrets')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='127.0.0.1', help='Host to listen on')
    server_parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    
    args = parser.parse_args()
    
    if args.command == 'server':
        print(f"Starting vault API server on {args.host}:{args.port}")
        start_api_server(args.host, args.port)
    else:
        # CLI operations
        vault = create_cli_app()
        
        if args.command == 'put':
            success = vault.put(args.key, args.value)
            if success:
                print(f"Stored secret '{args.key}' successfully")
            else:
                print(f"Failed to store secret '{args.key}'")
                sys.exit(1)
                
        elif args.command == 'get':
            value = vault.get(args.key)
            if value is not None:
                print(value)
            else:
                print(f"Secret '{args.key}' not found")
                sys.exit(1)
                
        elif args.command == 'delete':
            success = vault.delete(args.key)
            if success:
                print(f"Deleted secret '{args.key}' successfully")
            else:
                print(f"Secret '{args.key}' not found or could not be deleted")
                sys.exit(1)
                
        elif args.command == 'list':
            keys = vault.list_keys()
            if keys:
                print("Stored secrets:")
                for key in keys:
                    print(f"  - {key}")
            else:
                print("No secrets stored")
        else:
            parser.print_help()


if __name__ == '__main__':
    import sys
    main()
