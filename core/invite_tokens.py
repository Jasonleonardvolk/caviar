"""
Invite Token System - Secure token generation and validation
Handles both HMAC-signed tokens and human-readable codes
"""

import hmac
import hashlib
import json
import time
import secrets
import string
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import base64
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Get secret from environment or generate one
INVITE_SECRET = os.environ.get('TORI_INVITE_SECRET', secrets.token_hex(32))

@dataclass
class InviteToken:
    """Invite token data"""
    token_id: str
    group_id: str
    created_by: str
    created_at: float
    expires_at: Optional[float] = None
    max_uses: Optional[int] = None
    uses: int = 0
    is_single_use: bool = False
    human_code: Optional[str] = None  # Human-readable code
    
    def is_valid(self) -> bool:
        """Check if token is still valid"""
        # Check expiry
        if self.expires_at and time.time() > self.expires_at:
            return False
        
        # Check usage limit
        if self.max_uses and self.uses >= self.max_uses:
            return False
        
        # Single use check
        if self.is_single_use and self.uses > 0:
            return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "token_id": self.token_id,
            "group_id": self.group_id,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "max_uses": self.max_uses,
            "uses": self.uses,
            "is_single_use": self.is_single_use,
            "human_code": self.human_code
        }

class InviteTokenManager:
    """Manages invite tokens with persistence"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'InviteTokenManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.tokens: Dict[str, InviteToken] = {}
        self.human_codes: Dict[str, str] = {}  # human_code -> token_id mapping
        self.data_dir = Path("data/invites")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_file = self.data_dir / "tokens.json"
        
        # Load existing tokens
        self._load_tokens()
        
        logger.info(f"🎟️ InviteTokenManager initialized with {len(self.tokens)} tokens")
    
    def create_token(self, group_id: str, created_by: str, 
                    expires_in: Optional[int] = None,
                    max_uses: Optional[int] = None,
                    is_single_use: bool = False,
                    generate_human_code: bool = True) -> Tuple[str, InviteToken]:
        """
        Create a new invite token
        
        Args:
            group_id: The group this invite is for
            created_by: User ID creating the invite
            expires_in: Seconds until expiry (None = never expires)
            max_uses: Maximum number of uses (None = unlimited)
            is_single_use: If True, token can only be used once
            generate_human_code: If True, also generate a human-readable code
            
        Returns:
            Tuple of (signed_token, InviteToken object)
        """
        # Generate token ID
        token_id = secrets.token_urlsafe(16)
        
        # Calculate expiry
        expires_at = None
        if expires_in:
            expires_at = time.time() + expires_in
        
        # Generate human-readable code if requested
        human_code = None
        if generate_human_code:
            human_code = self._generate_human_code()
            # Ensure uniqueness
            while human_code in self.human_codes:
                human_code = self._generate_human_code()
        
        # Create token object
        token = InviteToken(
            token_id=token_id,
            group_id=group_id,
            created_by=created_by,
            created_at=time.time(),
            expires_at=expires_at,
            max_uses=max_uses,
            is_single_use=is_single_use,
            human_code=human_code
        )
        
        # Store token
        self.tokens[token_id] = token
        if human_code:
            self.human_codes[human_code] = token_id
        
        # Save to disk
        self._save_tokens()
        
        # Generate signed token
        signed_token = self._sign_token(token_id)
        
        logger.info(f"✅ Created invite token for group {group_id}")
        return signed_token, token
    
    def validate_token(self, token_or_code: str) -> Optional[InviteToken]:
        """
        Validate a token or human-readable code
        
        Args:
            token_or_code: Either a signed token or human-readable code
            
        Returns:
            InviteToken if valid, None otherwise
        """
        # First check if it's a human-readable code
        if len(token_or_code) <= 10 and token_or_code.upper() in self.human_codes:
            token_id = self.human_codes[token_or_code.upper()]
            token = self.tokens.get(token_id)
            if token and token.is_valid():
                return token
            return None
        
        # Try to verify as signed token
        token_id = self._verify_token(token_or_code)
        if not token_id:
            return None
        
        # Get token object
        token = self.tokens.get(token_id)
        if not token:
            return None
        
        # Check validity
        if not token.is_valid():
            return None
        
        return token
    
    def use_token(self, token_id: str) -> bool:
        """Mark a token as used"""
        token = self.tokens.get(token_id)
        if not token:
            return False
        
        token.uses += 1
        self._save_tokens()
        
        # Clean up if exhausted
        if not token.is_valid():
            self._cleanup_token(token_id)
        
        return True
    
    def revoke_token(self, token_id: str):
        """Revoke a token"""
        self._cleanup_token(token_id)
        logger.info(f"🚫 Revoked token {token_id}")
    
    def get_group_tokens(self, group_id: str) -> List[InviteToken]:
        """Get all tokens for a group"""
        return [t for t in self.tokens.values() if t.group_id == group_id]
    
    def _generate_human_code(self) -> str:
        """Generate a human-readable invite code"""
        # Generate 6-8 character code like "TORI-A3X9"
        prefix = "TORI"
        suffix_length = 4
        chars = string.ascii_uppercase + string.digits
        suffix = ''.join(secrets.choice(chars) for _ in range(suffix_length))
        return f"{prefix}-{suffix}"
    
    def _sign_token(self, token_id: str) -> str:
        """Generate HMAC-signed token"""
        # Create payload
        payload = {
            "token_id": token_id,
            "t": int(time.time())  # Timestamp
        }
        
        # Encode payload
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode().rstrip('=')
        
        # Generate signature
        signature = hmac.new(
            INVITE_SECRET.encode(),
            payload_b64.encode(),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')
        
        # Combine
        return f"{payload_b64}.{signature_b64}"
    
    def _verify_token(self, signed_token: str) -> Optional[str]:
        """Verify HMAC-signed token and return token_id"""
        try:
            # Split token
            parts = signed_token.split('.')
            if len(parts) != 2:
                return None
            
            payload_b64, signature_b64 = parts
            
            # Verify signature
            expected_signature = hmac.new(
                INVITE_SECRET.encode(),
                payload_b64.encode(),
                hashlib.sha256
            ).digest()
            
            # Decode provided signature
            provided_signature = base64.urlsafe_b64decode(signature_b64 + '==')
            
            # Compare
            if not hmac.compare_digest(expected_signature, provided_signature):
                return None
            
            # Decode payload
            payload_bytes = base64.urlsafe_b64decode(payload_b64 + '==')
            payload = json.loads(payload_bytes)
            
            return payload.get("token_id")
            
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def _cleanup_token(self, token_id: str):
        """Remove a token and its human code"""
        token = self.tokens.get(token_id)
        if token:
            if token.human_code:
                self.human_codes.pop(token.human_code, None)
            self.tokens.pop(token_id, None)
            self._save_tokens()
    
    def _save_tokens(self):
        """Save tokens to disk"""
        try:
            data = {
                "tokens": {tid: t.to_dict() for tid, t in self.tokens.items()},
                "human_codes": self.human_codes,
                "last_updated": time.time()
            }
            
            with open(self.tokens_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")
    
    def _load_tokens(self):
        """Load tokens from disk"""
        try:
            if self.tokens_file.exists():
                with open(self.tokens_file, 'r') as f:
                    data = json.load(f)
                
                # Load tokens
                for tid, tdata in data.get("tokens", {}).items():
                    self.tokens[tid] = InviteToken(**tdata)
                
                # Load human code mappings
                self.human_codes = data.get("human_codes", {})
                
                # Clean up expired tokens
                expired = []
                for tid, token in self.tokens.items():
                    if not token.is_valid():
                        expired.append(tid)
                
                for tid in expired:
                    self._cleanup_token(tid)
                
                logger.info(f"Loaded {len(self.tokens)} valid tokens")
                
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
    
    def cleanup_expired(self):
        """Clean up all expired tokens"""
        expired = []
        for tid, token in self.tokens.items():
            if not token.is_valid():
                expired.append(tid)
        
        for tid in expired:
            self._cleanup_token(tid)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired tokens")
