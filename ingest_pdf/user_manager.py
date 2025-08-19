"""
👤 TORI USER MANAGER - Authentication & Authorization
Production Ready: June 4, 2025
Features: JWT Authentication, Role-Based Access, Session Management
Security: Password hashing, Token validation, Rate limiting
"""

import os
import json
import jwt
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class UserRole(Enum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

@dataclass
class UserSession:
    user_id: str
    username: str
    email: str
    role: UserRole
    organization_ids: List[str]
    token: str
    expires_at: datetime
    created_at: datetime
    last_active: datetime

@dataclass
class LoginRequest:
    username: str
    password: str

@dataclass
class LoginResponse:
    success: bool
    token: Optional[str]
    user: Optional[Dict[str, Any]]
    message: str
    expires_at: Optional[datetime]

class UserManager:
    """
    👤 User Authentication & Authorization Manager
    
    Features:
    - JWT-based authentication
    - Role-based access control
    - Session management
    - Password security
    - Multi-organization support
    """
    
    def __init__(self, data_dir: str = "data", secret_key: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.users_dir = self.data_dir / "users"
        
        # JWT settings
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "tori-jwt-secret-key-change-in-production")
        self.token_expiry_hours = 24  # 24 hour token expiry
        self.algorithm = "HS256"
        
        # Rate limiting (simple in-memory for now)
        self.login_attempts = {}  # ip -> {'count': int, 'last_attempt': datetime}
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 15
        
        # Active sessions
        self.active_sessions = {}  # token -> UserSession
        
        # Ensure directories exist
        self.users_dir.mkdir(parents=True, exist_ok=True)
        self.users_file = self.users_dir / "users.json"
        
        logger.info("👤 User Manager initialized with JWT authentication")
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file with error handling"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}
    
    def _save_json(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Save JSON file with error handling"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")
            return False
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt using PBKDF2"""
        salt = os.urandom(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt.hex() + pwdhash.hex()
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt = bytes.fromhex(stored_hash[:64])
            stored_key = stored_hash[64:]
            pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return pwdhash.hex() == stored_key
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def _generate_jwt_token(self, user_id: str, username: str, role: str, organization_ids: List[str]) -> str:
        """Generate JWT token for user"""
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=self.token_expiry_hours)
        
        payload = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'organization_ids': organization_ids,
            'iat': now,
            'exp': expires_at,
            'iss': 'tori-system'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def _verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def _check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP address is rate limited"""
        now = datetime.utcnow()
        
        if ip_address in self.login_attempts:
            attempt_data = self.login_attempts[ip_address]
            
            # Reset if lockout period has passed
            time_since_last = now - attempt_data['last_attempt']
            if time_since_last > timedelta(minutes=self.lockout_duration_minutes):
                del self.login_attempts[ip_address]
                return True
            
            # Check if too many attempts
            if attempt_data['count'] >= self.max_login_attempts:
                logger.warning(f"Rate limit exceeded for IP: {ip_address}")
                return False
        
        return True
    
    def _record_login_attempt(self, ip_address: str, success: bool):
        """Record login attempt for rate limiting"""
        now = datetime.utcnow()
        
        if success:
            # Clear attempts on successful login
            if ip_address in self.login_attempts:
                del self.login_attempts[ip_address]
        else:
            # Increment failed attempts
            if ip_address not in self.login_attempts:
                self.login_attempts[ip_address] = {'count': 0, 'last_attempt': now}
            
            self.login_attempts[ip_address]['count'] += 1
            self.login_attempts[ip_address]['last_attempt'] = now
    
    # ===================================================================
    # USER AUTHENTICATION
    # ===================================================================
    
    def login(self, username: str, password: str, ip_address: str = "127.0.0.1") -> LoginResponse:
        """Authenticate user and create session"""
        try:
            # Check rate limiting
            if not self._check_rate_limit(ip_address):
                return LoginResponse(
                    success=False,
                    token=None,
                    user=None,
                    message=f"Too many login attempts. Please try again in {self.lockout_duration_minutes} minutes.",
                    expires_at=None
                )
            
            # Load users
            users_data = self._load_json(self.users_file)
            
            # Find user
            user_data = None
            for uid, udata in users_data.get("users", {}).items():
                if udata["username"] == username or udata["email"] == username:
                    user_data = udata
                    break
            
            if not user_data:
                self._record_login_attempt(ip_address, False)
                return LoginResponse(
                    success=False,
                    token=None,
                    user=None,
                    message="Invalid username or password",
                    expires_at=None
                )
            
            # Verify password
            if not self._verify_password(password, user_data["password_hash"]):
                self._record_login_attempt(ip_address, False)
                return LoginResponse(
                    success=False,
                    token=None,
                    user=None,
                    message="Invalid username or password",
                    expires_at=None
                )
            
            # Check if user is active
            if not user_data.get("is_active", True):
                self._record_login_attempt(ip_address, False)
                return LoginResponse(
                    success=False,
                    token=None,
                    user=None,
                    message="Account is disabled",
                    expires_at=None
                )
            
            # Generate JWT token
            token = self._generate_jwt_token(
                user_data["id"],
                user_data["username"],
                user_data["role"],
                user_data["organization_ids"]
            )
            
            # Create session
            now = datetime.utcnow()
            expires_at = now + timedelta(hours=self.token_expiry_hours)
            
            session = UserSession(
                user_id=user_data["id"],
                username=user_data["username"],
                email=user_data["email"],
                role=UserRole(user_data["role"]),
                organization_ids=user_data["organization_ids"],
                token=token,
                expires_at=expires_at,
                created_at=now,
                last_active=now
            )
            
            # Store session
            self.active_sessions[token] = session
            
            # Update user's last active
            user_data["last_active"] = now.isoformat()
            users_data["users"][user_data["id"]] = user_data
            self._save_json(self.users_file, users_data)
            
            # Record successful login
            self._record_login_attempt(ip_address, True)
            
            # Prepare user data for response (without sensitive info)
            user_response = {
                "id": user_data["id"],
                "username": user_data["username"],
                "email": user_data["email"],
                "role": user_data["role"],
                "organization_ids": user_data["organization_ids"],
                "created_at": user_data["created_at"],
                "last_active": user_data["last_active"],
                "concept_count": user_data.get("concept_count", 0)
            }
            
            logger.info(f"✅ User logged in: {username}")
            
            return LoginResponse(
                success=True,
                token=token,
                user=user_response,
                message="Login successful",
                expires_at=expires_at
            )
            
        except Exception as e:
            logger.error(f"Login error for {username}: {e}")
            self._record_login_attempt(ip_address, False)
            return LoginResponse(
                success=False,
                token=None,
                user=None,
                message="Internal server error",
                expires_at=None
            )
    
    def logout(self, token: str) -> bool:
        """Logout user and invalidate session"""
        try:
            if token in self.active_sessions:
                session = self.active_sessions[token]
                del self.active_sessions[token]
                logger.info(f"✅ User logged out: {session.username}")
                return True
            return False
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    def validate_token(self, token: str) -> Optional[UserSession]:
        """Validate JWT token and return session"""
        try:
            # Check if session exists
            if token not in self.active_sessions:
                # Try to verify token anyway (in case session was lost)
                payload = self._verify_jwt_token(token)
                if not payload:
                    return None
                
                # Recreate session from token
                session = UserSession(
                    user_id=payload["user_id"],
                    username=payload["username"],
                    email="",  # Will be filled from user data if needed
                    role=UserRole(payload["role"]),
                    organization_ids=payload["organization_ids"],
                    token=token,
                    expires_at=datetime.fromtimestamp(payload["exp"]),
                    created_at=datetime.fromtimestamp(payload["iat"]),
                    last_active=datetime.utcnow()
                )
                
                # Check if token is expired
                if session.expires_at < datetime.utcnow():
                    return None
                
                self.active_sessions[token] = session
            
            session = self.active_sessions[token]
            
            # Check if session is expired
            if session.expires_at < datetime.utcnow():
                del self.active_sessions[token]
                return None
            
            # Update last active
            session.last_active = datetime.utcnow()
            
            return session
            
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
    
    def get_current_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Get current user info from token"""
        session = self.validate_token(token)
        if not session:
            return None
        
        try:
            users_data = self._load_json(self.users_file)
            user_data = users_data.get("users", {}).get(session.user_id)
            
            if user_data:
                return {
                    "id": user_data["id"],
                    "username": user_data["username"],
                    "email": user_data["email"],
                    "role": user_data["role"],
                    "organization_ids": user_data["organization_ids"],
                    "created_at": user_data["created_at"],
                    "last_active": user_data["last_active"],
                    "concept_count": user_data.get("concept_count", 0),
                    "preferences": user_data.get("preferences", {})
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Get current user error: {e}")
            return None
    
    # ===================================================================
    # USER REGISTRATION
    # ===================================================================
    
    def register_user(self, username: str, email: str, password: str, 
                     role: UserRole = UserRole.MEMBER) -> Optional[Dict[str, Any]]:
        """Register a new user"""
        try:
            # Load existing users
            users_data = self._load_json(self.users_file)
            if "users" not in users_data:
                users_data = {
                    "users": {},
                    "metadata": {
                        "version": "1.0.0",
                        "created_at": datetime.utcnow().isoformat(),
                        "total_users": 0
                    }
                }
            
            # Check if user already exists
            for user_data in users_data["users"].values():
                if user_data["username"] == username or user_data["email"] == email:
                    logger.warning(f"User already exists: {username} / {email}")
                    return None
            
            # Create new user
            user_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            user_data = {
                "id": user_id,
                "username": username,
                "email": email,
                "password_hash": self._hash_password(password),
                "role": role.value,
                "organization_ids": [],
                "created_at": now.isoformat(),
                "last_active": now.isoformat(),
                "is_active": True,
                "preferences": {},
                "concept_count": 0
            }
            
            # Save user
            users_data["users"][user_id] = user_data
            users_data["metadata"]["total_users"] = len(users_data["users"])
            
            if self._save_json(self.users_file, users_data):
                # Create user's concept directory
                user_concepts_dir = self.users_dir / user_id
                user_concepts_dir.mkdir(exist_ok=True)
                
                # Initialize user's concept file
                user_concepts_file = user_concepts_dir / "concepts.json"
                initial_concepts = {
                    "concepts": {},
                    "metadata": {
                        "user_id": user_id,
                        "created_at": now.isoformat(),
                        "total_concepts": 0
                    }
                }
                self._save_json(user_concepts_file, initial_concepts)
                
                logger.info(f"✅ Registered user: {username} ({user_id})")
                
                # Return user data (without password hash)
                return {
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "role": role.value,
                    "organization_ids": [],
                    "created_at": user_data["created_at"],
                    "last_active": user_data["last_active"]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Registration error for {username}: {e}")
            return None
    
    # ===================================================================
    # AUTHORIZATION HELPERS
    # ===================================================================
    
    def check_permission(self, token: str, required_role: UserRole = UserRole.MEMBER) -> bool:
        """Check if user has required permission level"""
        session = self.validate_token(token)
        if not session:
            return False
        
        role_hierarchy = {
            UserRole.VIEWER: 1,
            UserRole.MEMBER: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(session.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level
    
    def is_admin(self, token: str) -> bool:
        """Check if user is admin"""
        return self.check_permission(token, UserRole.ADMIN)
    
    def can_access_organization(self, token: str, organization_id: str) -> bool:
        """Check if user can access specific organization"""
        session = self.validate_token(token)
        if not session:
            return False
        
        return organization_id in session.organization_ids or session.role == UserRole.ADMIN
    
    # ===================================================================
    # SESSION MANAGEMENT
    # ===================================================================
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions (admin only)"""
        try:
            sessions = []
            now = datetime.utcnow()
            
            # Clean expired sessions
            expired_tokens = []
            for token, session in self.active_sessions.items():
                if session.expires_at < now:
                    expired_tokens.append(token)
                else:
                    sessions.append({
                        "user_id": session.user_id,
                        "username": session.username,
                        "role": session.role.value,
                        "created_at": session.created_at.isoformat(),
                        "last_active": session.last_active.isoformat(),
                        "expires_at": session.expires_at.isoformat()
                    })
            
            # Remove expired sessions
            for token in expired_tokens:
                del self.active_sessions[token]
            
            return sessions
            
        except Exception as e:
            logger.error(f"Get active sessions error: {e}")
            return []
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            now = datetime.utcnow()
            expired_tokens = []
            
            for token, session in self.active_sessions.items():
                if session.expires_at < now:
                    expired_tokens.append(token)
            
            for token in expired_tokens:
                del self.active_sessions[token]
            
            if expired_tokens:
                logger.info(f"🧹 Cleaned up {len(expired_tokens)} expired sessions")
            
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
    
    # ===================================================================
    # USER MANAGEMENT
    # ===================================================================
    
    def get_all_users(self, admin_token: str) -> Optional[List[Dict[str, Any]]]:
        """Get all users (admin only)"""
        if not self.is_admin(admin_token):
            logger.warning("Unauthorized attempt to get all users")
            return None
        
        try:
            users_data = self._load_json(self.users_file)
            users = []
            
            for user_data in users_data.get("users", {}).values():
                users.append({
                    "id": user_data["id"],
                    "username": user_data["username"],
                    "email": user_data["email"],
                    "role": user_data["role"],
                    "organization_ids": user_data["organization_ids"],
                    "created_at": user_data["created_at"],
                    "last_active": user_data["last_active"],
                    "is_active": user_data.get("is_active", True),
                    "concept_count": user_data.get("concept_count", 0)
                })
            
            return users
            
        except Exception as e:
            logger.error(f"Get all users error: {e}")
            return None
    
    def update_user_role(self, admin_token: str, user_id: str, new_role: UserRole) -> bool:
        """Update user role (admin only)"""
        if not self.is_admin(admin_token):
            logger.warning("Unauthorized attempt to update user role")
            return False
        
        try:
            users_data = self._load_json(self.users_file)
            
            if user_id in users_data.get("users", {}):
                users_data["users"][user_id]["role"] = new_role.value
                
                if self._save_json(self.users_file, users_data):
                    logger.info(f"✅ Updated user {user_id} role to {new_role.value}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Update user role error: {e}")
            return False

# ===================================================================
# GLOBAL INSTANCE
# ===================================================================

# Global instance for easy access
_user_manager = None

def get_user_manager() -> UserManager:
    """Get or create global user manager instance"""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager

if __name__ == "__main__":
    # Demo/test functionality
    print("👤 TORI User Manager Demo")
    
    um = UserManager()
    
    # Test user registration
    user = um.register_user("demo_user", "demo@example.com", "password123", UserRole.MEMBER)
    if user:
        print(f"Registered user: {user['username']}")
    
    # Test login
    login_result = um.login("demo_user", "password123")
    if login_result.success:
        print(f"Login successful! Token: {login_result.token[:20]}...")
        
        # Test token validation
        session = um.validate_token(login_result.token)
        if session:
            print(f"Token valid for user: {session.username}")
        
        # Test logout
        logout_success = um.logout(login_result.token)
        print(f"Logout successful: {logout_success}")
    
    print("✅ User Manager test complete!")
