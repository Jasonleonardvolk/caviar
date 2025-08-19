"""
🏢 TORI MULTI-TENANT MANAGER - Complete Three-Tier Architecture
Production Ready: June 4, 2025
Features: User Management, Organization Support, Three-Tier Knowledge System
Architecture: Private → Organization → Foundation
"""

import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class UserRole(Enum):
    ADMIN = "admin"
    MEMBER = "member" 
    VIEWER = "viewer"

class KnowledgeTier(Enum):
    PRIVATE = "private"
    ORGANIZATION = "organization"
    FOUNDATION = "foundation"

@dataclass
class User:
    id: str
    username: str
    email: str
    password_hash: str
    organization_ids: List[str]
    role: UserRole
    created_at: str
    last_active: str
    preferences: Dict[str, Any]
    concept_count: int = 0
    is_active: bool = True

@dataclass  
class Organization:
    id: str
    name: str
    description: str
    admin_user_ids: List[str]
    member_user_ids: List[str]
    viewer_user_ids: List[str]
    created_at: str
    settings: Dict[str, Any]
    concept_count: int = 0
    is_active: bool = True

@dataclass
class ConceptMetadata:
    id: str
    name: str
    tier: KnowledgeTier
    owner_id: str  # user_id or org_id or 'foundation'
    created_at: str
    updated_at: str
    access_count: int
    confidence: float
    tags: List[str]
    source_document: Optional[str] = None

class MultiTenantManager:
    """
    🏢 Core Multi-Tenant Management System
    
    Manages the three-tier knowledge architecture:
    1. Foundation Layer: Global admin knowledge (Darwin, AI/ML, etc.)
    2. Organization Layer: Company/team specific knowledge
    3. Private Layer: Individual user knowledge
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.users_dir = self.data_dir / "users"
        self.orgs_dir = self.data_dir / "organizations"  
        self.foundation_dir = self.data_dir / "foundation"
        
        # Ensure directories exist
        for dir_path in [self.users_dir, self.orgs_dir, self.foundation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.users_file = self.users_dir / "users.json"
        self.organizations_file = self.orgs_dir / "organizations.json"
        self.foundation_concepts_file = self.foundation_dir / "concepts.json"
        
        # Initialize storage files
        self._initialize_storage()
        
        logger.info("🏢 Multi-Tenant Manager initialized with three-tier architecture")
    
    def _initialize_storage(self):
        """Initialize storage files if they don't exist"""
        
        # Initialize users file
        if not self.users_file.exists():
            initial_users = {
                "users": {},
                "metadata": {
                    "version": "1.0.0",
                    "created_at": datetime.now().isoformat(),
                    "total_users": 0
                }
            }
            self._save_json(self.users_file, initial_users)
            logger.info("✅ Initialized users.json")
        
        # Initialize organizations file
        if not self.organizations_file.exists():
            initial_orgs = {
                "organizations": {},
                "metadata": {
                    "version": "1.0.0", 
                    "created_at": datetime.now().isoformat(),
                    "total_organizations": 0
                }
            }
            self._save_json(self.organizations_file, initial_orgs)
            logger.info("✅ Initialized organizations.json")
        
        # Initialize foundation concepts
        if not self.foundation_concepts_file.exists():
            foundation_concepts = self._create_foundation_knowledge()
            self._save_json(self.foundation_concepts_file, foundation_concepts)
            logger.info("✅ Initialized foundation concepts with your core knowledge")
    
    def _create_foundation_knowledge(self) -> Dict[str, Any]:
        """Create foundation knowledge base with your core concepts"""
        now = datetime.now().isoformat()
        
        foundation_concepts = {
            "concepts": {
                # Darwin & Evolution
                "evolution": {
                    "id": "found_evolution",
                    "name": "Evolution",
                    "confidence": 0.95,
                    "context": "Charles Darwin's theory of evolution by natural selection - the fundamental mechanism of biological change",
                    "tier": "foundation",
                    "owner_id": "foundation",
                    "created_at": now,
                    "updated_at": now,
                    "access_count": 0,
                    "tags": ["darwin", "biology", "science", "theory"],
                    "domain": "biology"
                },
                "natural_selection": {
                    "id": "found_natural_selection", 
                    "name": "Natural Selection",
                    "confidence": 0.93,
                    "context": "Darwin's mechanism explaining how species evolve through differential survival and reproduction",
                    "tier": "foundation",
                    "owner_id": "foundation", 
                    "created_at": now,
                    "updated_at": now,
                    "access_count": 0,
                    "tags": ["darwin", "evolution", "biology", "mechanism"],
                    "domain": "biology"
                },
                "species": {
                    "id": "found_species",
                    "name": "Species",
                    "confidence": 0.90,
                    "context": "Biological classification unit and the focus of evolutionary processes",
                    "tier": "foundation", 
                    "owner_id": "foundation",
                    "created_at": now,
                    "updated_at": now,
                    "access_count": 0,
                    "tags": ["biology", "classification", "evolution"],
                    "domain": "biology"
                },
                
                # AI & Machine Learning
                "artificial_intelligence": {
                    "id": "found_ai",
                    "name": "Artificial Intelligence", 
                    "confidence": 0.92,
                    "context": "The simulation of human intelligence in machines that are programmed to think and learn",
                    "tier": "foundation",
                    "owner_id": "foundation",
                    "created_at": now,
                    "updated_at": now, 
                    "access_count": 0,
                    "tags": ["ai", "technology", "intelligence", "machine"],
                    "domain": "technology"
                },
                "machine_learning": {
                    "id": "found_ml",
                    "name": "Machine Learning",
                    "confidence": 0.91,
                    "context": "A subset of AI that enables machines to learn and improve from experience without explicit programming",
                    "tier": "foundation",
                    "owner_id": "foundation",
                    "created_at": now,
                    "updated_at": now,
                    "access_count": 0,
                    "tags": ["ml", "ai", "algorithms", "learning"],
                    "domain": "technology"
                },
                "neural_networks": {
                    "id": "found_neural_nets",
                    "name": "Neural Networks",
                    "confidence": 0.89,
                    "context": "Computing systems inspired by biological neural networks that constitute animal brains",
                    "tier": "foundation", 
                    "owner_id": "foundation",
                    "created_at": now,
                    "updated_at": now,
                    "access_count": 0,
                    "tags": ["neural", "networks", "ai", "deep_learning"],
                    "domain": "technology"
                },
                
                # Business & Strategy
                "strategic_planning": {
                    "id": "found_strategy",
                    "name": "Strategic Planning",
                    "confidence": 0.88,
                    "context": "The process of defining organizational direction and making decisions on allocating resources",
                    "tier": "foundation",
                    "owner_id": "foundation", 
                    "created_at": now,
                    "updated_at": now,
                    "access_count": 0,
                    "tags": ["strategy", "business", "planning", "management"],
                    "domain": "business"
                },
                "competitive_analysis": {
                    "id": "found_competitive",
                    "name": "Competitive Analysis", 
                    "confidence": 0.87,
                    "context": "Assessment of strengths and weaknesses of current and potential competitors",
                    "tier": "foundation",
                    "owner_id": "foundation",
                    "created_at": now,
                    "updated_at": now,
                    "access_count": 0,
                    "tags": ["competition", "analysis", "market", "strategy"],
                    "domain": "business"
                },
                
                # Physics & Mathematics
                "quantum_mechanics": {
                    "id": "found_quantum",
                    "name": "Quantum Mechanics",
                    "confidence": 0.86,
                    "context": "Fundamental theory in physics describing physical properties of nature at atomic and subatomic scales",
                    "tier": "foundation",
                    "owner_id": "foundation",
                    "created_at": now,
                    "updated_at": now,
                    "access_count": 0,
                    "tags": ["quantum", "physics", "mechanics", "theory"],
                    "domain": "physics"
                },
                "mathematics": {
                    "id": "found_mathematics",
                    "name": "Mathematics",
                    "confidence": 0.85,
                    "context": "The abstract science of number, quantity, and space, either as abstract concepts or as applied",
                    "tier": "foundation",
                    "owner_id": "foundation", 
                    "created_at": now,
                    "updated_at": now,
                    "access_count": 0,
                    "tags": ["math", "science", "abstract", "numbers"],
                    "domain": "mathematics"
                }
            },
            "metadata": {
                "version": "1.0.0",
                "created_at": now,
                "total_concepts": 10,
                "description": "Foundation knowledge base with core concepts",
                "domains": ["biology", "technology", "business", "physics", "mathematics"]
            }
        }
        
        return foundation_concepts
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}
    
    def _save_json(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Save JSON file with error handling"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")
            return False
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
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
        except:
            return False
    
    # ===================================================================
    # USER MANAGEMENT
    # ===================================================================
    
    def create_user(self, username: str, email: str, password: str, 
                   organization_ids: List[str] = None, role: UserRole = UserRole.MEMBER) -> Optional[User]:
        """Create a new user"""
        try:
            users_data = self._load_json(self.users_file)
            
            # Check if user already exists
            for user_data in users_data["users"].values():
                if user_data["username"] == username or user_data["email"] == email:
                    logger.warning(f"User already exists: {username} / {email}")
                    return None
            
            # Create user
            user_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            user = User(
                id=user_id,
                username=username,
                email=email,
                password_hash=self._hash_password(password),
                organization_ids=organization_ids or [],
                role=role,
                created_at=now,
                last_active=now,
                preferences={},
                concept_count=0
            )
            
            # Save user
            users_data["users"][user_id] = asdict(user)
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
                        "created_at": now,
                        "total_concepts": 0
                    }
                }
                self._save_json(user_concepts_file, initial_concepts)
                
                logger.info(f"✅ Created user: {username} ({user_id})")
                return user
            
        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user by username/password"""
        try:
            users_data = self._load_json(self.users_file)
            
            for user_data in users_data["users"].values():
                if (user_data["username"] == username or user_data["email"] == username):
                    if self._verify_password(password, user_data["password_hash"]):
                        # Update last active
                        user_data["last_active"] = datetime.now().isoformat()
                        self._save_json(self.users_file, users_data)
                        
                        user = User(**user_data)
                        logger.info(f"✅ Authenticated user: {username}")
                        return user
            
            logger.warning(f"Authentication failed for: {username}")
            return None
            
        except Exception as e:
            logger.error(f"Authentication error for {username}: {e}")
            return None
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            users_data = self._load_json(self.users_file)
            user_data = users_data["users"].get(user_id)
            
            if user_data:
                return User(**user_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None
    
    def update_user_activity(self, user_id: str):
        """Update user's last active timestamp"""
        try:
            users_data = self._load_json(self.users_file)
            if user_id in users_data["users"]:
                users_data["users"][user_id]["last_active"] = datetime.now().isoformat()
                self._save_json(self.users_file, users_data)
        except Exception as e:
            logger.error(f"Failed to update user activity {user_id}: {e}")
    
    # ===================================================================
    # ORGANIZATION MANAGEMENT  
    # ===================================================================
    
    def create_organization(self, name: str, description: str, admin_user_id: str) -> Optional[Organization]:
        """Create a new organization"""
        try:
            orgs_data = self._load_json(self.organizations_file)
            
            # Check if organization already exists
            for org_data in orgs_data["organizations"].values():
                if org_data["name"] == name:
                    logger.warning(f"Organization already exists: {name}")
                    return None
            
            # Create organization
            org_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            organization = Organization(
                id=org_id,
                name=name,
                description=description,
                admin_user_ids=[admin_user_id],
                member_user_ids=[],
                viewer_user_ids=[],
                created_at=now,
                settings={},
                concept_count=0
            )
            
            # Save organization
            orgs_data["organizations"][org_id] = asdict(organization)
            orgs_data["metadata"]["total_organizations"] = len(orgs_data["organizations"])
            
            if self._save_json(self.organizations_file, orgs_data):
                # Create organization's concept directory
                org_concepts_dir = self.orgs_dir / org_id
                org_concepts_dir.mkdir(exist_ok=True)
                
                # Initialize organization's concept file
                org_concepts_file = org_concepts_dir / "concepts.json"
                initial_concepts = {
                    "concepts": {},
                    "metadata": {
                        "organization_id": org_id,
                        "created_at": now,
                        "total_concepts": 0
                    }
                }
                self._save_json(org_concepts_file, initial_concepts)
                
                # Add user to organization
                self._add_user_to_organization(admin_user_id, org_id)
                
                logger.info(f"✅ Created organization: {name} ({org_id})")
                return organization
                
        except Exception as e:
            logger.error(f"Failed to create organization {name}: {e}")
            return None
    
    def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID"""
        try:
            orgs_data = self._load_json(self.organizations_file)
            org_data = orgs_data["organizations"].get(org_id)
            
            if org_data:
                return Organization(**org_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get organization {org_id}: {e}")
            return None
    
    def _add_user_to_organization(self, user_id: str, org_id: str):
        """Add user to organization membership"""
        try:
            users_data = self._load_json(self.users_file)
            if user_id in users_data["users"]:
                if org_id not in users_data["users"][user_id]["organization_ids"]:
                    users_data["users"][user_id]["organization_ids"].append(org_id)
                    self._save_json(self.users_file, users_data)
        except Exception as e:
            logger.error(f"Failed to add user {user_id} to org {org_id}: {e}")
    
    def get_user_organizations(self, user_id: str) -> List[Organization]:
        """Get all organizations for a user"""
        try:
            user = self.get_user(user_id)
            if not user:
                return []
            
            organizations = []
            for org_id in user.organization_ids:
                org = self.get_organization(org_id)
                if org:
                    organizations.append(org)
            
            return organizations
            
        except Exception as e:
            logger.error(f"Failed to get organizations for user {user_id}: {e}")
            return []
    
    # ===================================================================
    # SYSTEM STATUS & STATISTICS
    # ===================================================================
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            users_data = self._load_json(self.users_file)
            orgs_data = self._load_json(self.organizations_file)
            foundation_data = self._load_json(self.foundation_concepts_file)
            
            total_users = len(users_data.get("users", {}))
            total_orgs = len(orgs_data.get("organizations", {}))
            foundation_concepts = len(foundation_data.get("concepts", {}))
            
            # Count total private concepts
            private_concepts = 0
            for user_dir in self.users_dir.iterdir():
                if user_dir.is_dir() and user_dir.name != "users.json":
                    concepts_file = user_dir / "concepts.json"
                    if concepts_file.exists():
                        user_concepts = self._load_json(concepts_file)
                        private_concepts += len(user_concepts.get("concepts", {}))
            
            # Count total organization concepts
            org_concepts = 0
            for org_dir in self.orgs_dir.iterdir():
                if org_dir.is_dir():
                    concepts_file = org_dir / "concepts.json"
                    if concepts_file.exists():
                        org_concepts_data = self._load_json(concepts_file)
                        org_concepts += len(org_concepts_data.get("concepts", {}))
            
            return {
                "system": {
                    "status": "operational",
                    "version": "1.0.0",
                    "architecture": "three_tier_multi_tenant"
                },
                "users": {
                    "total": total_users,
                    "active_users": total_users  # Simplified for now
                },
                "organizations": {
                    "total": total_orgs
                },
                "concepts": {
                    "foundation": foundation_concepts,
                    "organization": org_concepts,
                    "private": private_concepts,
                    "total": foundation_concepts + org_concepts + private_concepts
                },
                "knowledge_tiers": {
                    "foundation": {"concepts": foundation_concepts, "description": "Global admin knowledge"},
                    "organization": {"concepts": org_concepts, "description": "Company/team knowledge"},
                    "private": {"concepts": private_concepts, "description": "Individual user knowledge"}
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            health = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "checks": {}
            }
            
            # Check directory structure
            health["checks"]["directories"] = {
                "users": self.users_dir.exists(),
                "organizations": self.orgs_dir.exists(), 
                "foundation": self.foundation_dir.exists()
            }
            
            # Check core files
            health["checks"]["core_files"] = {
                "users.json": self.users_file.exists(),
                "organizations.json": self.organizations_file.exists(),
                "foundation_concepts.json": self.foundation_concepts_file.exists()
            }
            
            # Check if all core files are readable
            health["checks"]["file_access"] = {}
            for file_name, file_path in [
                ("users", self.users_file),
                ("organizations", self.organizations_file), 
                ("foundation", self.foundation_concepts_file)
            ]:
                try:
                    data = self._load_json(file_path)
                    health["checks"]["file_access"][file_name] = len(data) > 0
                except:
                    health["checks"]["file_access"][file_name] = False
            
            # Overall health determination
            all_checks = []
            for check_group in health["checks"].values():
                if isinstance(check_group, dict):
                    all_checks.extend(check_group.values())
                else:
                    all_checks.append(check_group)
            
            if all(all_checks):
                health["status"] = "healthy"
            else:
                health["status"] = "degraded"
            
            return health
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# ===================================================================
# GLOBAL INSTANCE
# ===================================================================

# Global instance for easy access
_multi_tenant_manager = None

def get_multi_tenant_manager() -> MultiTenantManager:
    """Get or create global multi-tenant manager instance"""
    global _multi_tenant_manager
    if _multi_tenant_manager is None:
        _multi_tenant_manager = MultiTenantManager()
    return _multi_tenant_manager

if __name__ == "__main__":
    # Demo/test functionality
    print("🏢 TORI Multi-Tenant Manager Demo")
    
    manager = MultiTenantManager()
    
    # Test system health
    health = manager.health_check()
    print(f"System Health: {health['status']}")
    
    # Test user creation
    user = manager.create_user("demo_user", "demo@example.com", "password123")
    if user:
        print(f"Created user: {user.username}")
    
    # Test authentication
    auth_user = manager.authenticate_user("demo_user", "password123")
    if auth_user:
        print(f"Authenticated: {auth_user.username}")
    
    # Get system stats
    stats = manager.get_system_stats()
    print(f"System Stats: {stats}")
    
    print("✅ Multi-Tenant Manager test complete!")
