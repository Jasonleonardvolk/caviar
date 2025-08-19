"""
Authentication system for ELFIN package registry.

This module provides tools for authenticating with the ELFIN package registry,
including token management, user authentication, and permission handling.
"""

import os
import json
import time
import base64
import hashlib
import logging
import getpass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
import requests

logger = logging.getLogger(__name__)


@dataclass
class User:
    """Represents a user in the authentication system."""
    username: str
    email: str
    full_name: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    orgs: List[str] = field(default_factory=list)
    

@dataclass
class Token:
    """Represents an authentication token."""
    token: str
    username: str
    created_at: float
    expires_at: Optional[float] = None
    scopes: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class AuthError(Exception):
    """Exception raised for authentication errors."""
    pass


class AuthManager:
    """
    Manages authentication with the ELFIN package registry.
    
    This provides a more comprehensive authentication system than just using
    API tokens directly, including user management, token refresh, and permission
    checking.
    """
    
    def __init__(self, registry_url: str = None, auth_file: Optional[Path] = None):
        """
        Initialize auth manager.
        
        Args:
            registry_url: URL of the registry, defaults to ELFIN_REGISTRY_URL env var
            auth_file: Path to the auth file, defaults to ~/.elfin/auth.json
        """
        self.registry_url = registry_url or os.environ.get('ELFIN_REGISTRY_URL', 'https://registry.elfin.dev')
        self.auth_file = auth_file or Path.home() / '.elfin' / 'auth.json'
        self._tokens: Dict[str, Token] = {}
        self._current_user: Optional[User] = None
        
        # Load existing tokens
        self._load_tokens()
    
    def _load_tokens(self) -> None:
        """Load tokens from the auth file."""
        if not self.auth_file.exists():
            return
        
        try:
            with open(self.auth_file, 'r') as f:
                data = json.load(f)
            
            for registry, token_data in data.get('tokens', {}).items():
                if registry == self.registry_url or registry == '*':
                    self._tokens[registry] = Token(
                        token=token_data.get('token', ''),
                        username=token_data.get('username', ''),
                        created_at=token_data.get('created_at', time.time()),
                        expires_at=token_data.get('expires_at'),
                        scopes=token_data.get('scopes', [])
                    )
        except Exception as e:
            logger.warning(f"Failed to load auth tokens: {e}")
    
    def _save_tokens(self) -> None:
        """Save tokens to the auth file."""
        try:
            # Ensure directory exists
            self.auth_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare token data
            token_data = {}
            for registry, token in self._tokens.items():
                token_data[registry] = {
                    'token': token.token,
                    'username': token.username,
                    'created_at': token.created_at,
                    'expires_at': token.expires_at,
                    'scopes': token.scopes
                }
            
            # Write to file
            with open(self.auth_file, 'w') as f:
                json.dump({'tokens': token_data}, f)
        except Exception as e:
            logger.warning(f"Failed to save auth tokens: {e}")
    
    def get_token(self) -> Optional[str]:
        """
        Get the current authentication token.
        
        Returns:
            Authentication token if available and not expired, None otherwise
        """
        token = self._tokens.get(self.registry_url) or self._tokens.get('*')
        if token and not token.is_expired:
            return token.token
        return None
    
    def set_token(self, token: str, username: str, expires_in: Optional[int] = None) -> None:
        """
        Set the authentication token.
        
        Args:
            token: Token string
            username: Username associated with the token
            expires_in: Token expiration time in seconds (from now), or None for no expiration
        """
        expires_at = time.time() + expires_in if expires_in else None
        self._tokens[self.registry_url] = Token(
            token=token,
            username=username,
            created_at=time.time(),
            expires_at=expires_at,
            scopes=['read', 'write']
        )
        self._save_tokens()
    
    def clear_token(self) -> None:
        """Clear the current authentication token."""
        if self.registry_url in self._tokens:
            del self._tokens[self.registry_url]
            self._save_tokens()
    
    def login(self, username: str, password: str) -> bool:
        """
        Log in to the registry.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            True if login successful, False otherwise
            
        Raises:
            AuthError: If login fails
        """
        try:
            response = requests.post(
                f"{self.registry_url}/api/v1/auth/login",
                json={'username': username, 'password': password}
            )
            
            if response.status_code != 200:
                raise AuthError(f"Login failed: {response.text}")
            
            data = response.json()
            token = data.get('token')
            expires_in = data.get('expires_in')
            
            if not token:
                raise AuthError("No token received from server")
            
            self.set_token(token, username, expires_in)
            return True
            
        except requests.RequestException as e:
            raise AuthError(f"Login request failed: {e}")
    
    def login_interactive(self) -> bool:
        """
        Interactive login prompt.
        
        Returns:
            True if login successful, False otherwise
        """
        try:
            username = input("Username: ")
            password = getpass.getpass("Password: ")
            
            return self.login(username, password)
        
        except (KeyboardInterrupt, EOFError):
            print("\nLogin canceled")
            return False
        
        except AuthError as e:
            print(f"Login failed: {e}")
            return False
    
    def logout(self) -> bool:
        """
        Log out from the registry.
        
        Returns:
            True if logout successful, False otherwise
        """
        try:
            token = self.get_token()
            if not token:
                # Already logged out
                return True
            
            headers = {'Authorization': f"Token {token}"}
            response = requests.post(
                f"{self.registry_url}/api/v1/auth/logout",
                headers=headers
            )
            
            # Clear local token regardless of server response
            self.clear_token()
            
            return response.status_code == 200
            
        except requests.RequestException as e:
            logger.warning(f"Logout request failed: {e}")
            # Still clear local token
            self.clear_token()
            return False
    
    def register(self, username: str, email: str, password: str, full_name: Optional[str] = None) -> bool:
        """
        Register a new user.
        
        Args:
            username: Username
            email: Email address
            password: Password
            full_name: Full name (optional)
            
        Returns:
            True if registration successful, False otherwise
            
        Raises:
            AuthError: If registration fails
        """
        try:
            payload = {
                'username': username,
                'email': email,
                'password': password
            }
            
            if full_name:
                payload['full_name'] = full_name
            
            response = requests.post(
                f"{self.registry_url}/api/v1/auth/register",
                json=payload
            )
            
            if response.status_code != 201:
                raise AuthError(f"Registration failed: {response.text}")
            
            # Auto-login after successful registration
            return self.login(username, password)
            
        except requests.RequestException as e:
            raise AuthError(f"Registration request failed: {e}")
    
    def get_current_user(self) -> Optional[User]:
        """
        Get the currently logged in user.
        
        Returns:
            User object if logged in, None otherwise
        """
        if self._current_user:
            return self._current_user
        
        token = self.get_token()
        if not token:
            return None
        
        try:
            headers = {'Authorization': f"Token {token}"}
            response = requests.get(
                f"{self.registry_url}/api/v1/auth/me",
                headers=headers
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            self._current_user = User(
                username=data.get('username', ''),
                email=data.get('email', ''),
                full_name=data.get('full_name'),
                permissions=data.get('permissions', []),
                orgs=data.get('orgs', [])
            )
            
            return self._current_user
            
        except requests.RequestException:
            return None
    
    def has_permission(self, permission: str) -> bool:
        """
        Check if the current user has a specific permission.
        
        Args:
            permission: Permission string
            
        Returns:
            True if user has permission, False otherwise
        """
        user = self.get_current_user()
        if not user:
            return False
        
        return permission in user.permissions
    
    def create_token(self, name: str, expires_in: Optional[int] = None) -> str:
        """
        Create a new API token.
        
        Args:
            name: Token name
            expires_in: Token expiration time in seconds, or None for no expiration
            
        Returns:
            The new token string
            
        Raises:
            AuthError: If token creation fails
        """
        token = self.get_token()
        if not token:
            raise AuthError("Not logged in")
        
        try:
            headers = {'Authorization': f"Token {token}"}
            payload = {'name': name}
            
            if expires_in:
                payload['expires_in'] = expires_in
            
            response = requests.post(
                f"{self.registry_url}/api/v1/auth/tokens",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 201:
                raise AuthError(f"Token creation failed: {response.text}")
            
            data = response.json()
            return data.get('token', '')
            
        except requests.RequestException as e:
            raise AuthError(f"Token creation request failed: {e}")
    
    def list_tokens(self) -> List[Dict[str, Any]]:
        """
        List all API tokens for the current user.
        
        Returns:
            List of token information
            
        Raises:
            AuthError: If listing tokens fails
        """
        token = self.get_token()
        if not token:
            raise AuthError("Not logged in")
        
        try:
            headers = {'Authorization': f"Token {token}"}
            response = requests.get(
                f"{self.registry_url}/api/v1/auth/tokens",
                headers=headers
            )
            
            if response.status_code != 200:
                raise AuthError(f"Listing tokens failed: {response.text}")
            
            return response.json().get('tokens', [])
            
        except requests.RequestException as e:
            raise AuthError(f"Listing tokens request failed: {e}")
    
    def revoke_token(self, token_id: str) -> bool:
        """
        Revoke an API token.
        
        Args:
            token_id: Token ID to revoke
            
        Returns:
            True if token revoked successfully, False otherwise
            
        Raises:
            AuthError: If revoking token fails
        """
        token = self.get_token()
        if not token:
            raise AuthError("Not logged in")
        
        try:
            headers = {'Authorization': f"Token {token}"}
            response = requests.delete(
                f"{self.registry_url}/api/v1/auth/tokens/{token_id}",
                headers=headers
            )
            
            return response.status_code == 204
            
        except requests.RequestException as e:
            raise AuthError(f"Revoking token request failed: {e}")
    
    def refresh_token(self) -> bool:
        """
        Refresh the current authentication token.
        
        Returns:
            True if token refreshed successfully, False otherwise
        """
        token = self.get_token()
        if not token:
            return False
        
        try:
            headers = {'Authorization': f"Token {token}"}
            response = requests.post(
                f"{self.registry_url}/api/v1/auth/refresh",
                headers=headers
            )
            
            if response.status_code != 200:
                return False
            
            data = response.json()
            new_token = data.get('token')
            expires_in = data.get('expires_in')
            
            if not new_token:
                return False
            
            # Update token
            current_token = self._tokens[self.registry_url]
            self.set_token(new_token, current_token.username, expires_in)
            return True
            
        except requests.RequestException:
            return False


def get_auth_manager(registry_url: Optional[str] = None) -> AuthManager:
    """
    Get an auth manager instance.
    
    Args:
        registry_url: Registry URL, defaults to ELFIN_REGISTRY_URL env var
        
    Returns:
        AuthManager instance
    """
    return AuthManager(registry_url)
