"""
Group Manager - Core group data model and operations
Handles group CRUD, membership management, and persistence
"""

import json
import logging
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Group:
    """Group data model"""
    id: str
    name: str
    owner_id: str
    created_at: float = field(default_factory=time.time)
    members: List[str] = field(default_factory=list)
    admins: List[str] = field(default_factory=list)  # Additional admins besides owner
    banned_users: Set[str] = field(default_factory=set)
    settings: Dict[str, any] = field(default_factory=dict)
    memory_handle: Optional[str] = None  # Reference to group's concept mesh
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "owner_id": self.owner_id,
            "created_at": self.created_at,
            "members": self.members,
            "admins": self.admins,
            "banned_users": list(self.banned_users),
            "settings": self.settings,
            "memory_handle": self.memory_handle
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Group':
        """Create from dictionary"""
        data = data.copy()
        data['banned_users'] = set(data.get('banned_users', []))
        return cls(**data)
    
    def is_admin(self, user_id: str) -> bool:
        """Check if user is admin (owner or in admins list)"""
        return user_id == self.owner_id or user_id in self.admins
    
    def is_member(self, user_id: str) -> bool:
        """Check if user is a member"""
        return user_id in self.members
    
    def is_banned(self, user_id: str) -> bool:
        """Check if user is banned"""
        return user_id in self.banned_users

class GroupManager:
    """Manages groups with thread-safe operations"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'GroupManager':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.groups: Dict[str, Group] = {}
        self.user_groups: Dict[str, Set[str]] = {}  # user_id -> set of group_ids
        self.data_dir = Path("data/groups")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.groups_file = self.data_dir / "groups.json"
        
        # Load existing groups
        self._load_groups()
        
        logger.info(f"👥 GroupManager initialized with {len(self.groups)} groups")
    
    def create_group(self, group_id: str, name: str, owner_id: str, 
                    initial_members: Optional[List[str]] = None) -> Group:
        """Create a new group"""
        with self._lock:
            # Check if group already exists
            if group_id in self.groups:
                raise ValueError(f"Group {group_id} already exists")
            
            # Create group
            members = [owner_id]
            if initial_members:
                # Add initial members (deduplicate)
                for member in initial_members:
                    if member not in members:
                        members.append(member)
            
            group = Group(
                id=group_id,
                name=name,
                owner_id=owner_id,
                members=members,
                memory_handle=f"group_mesh_{group_id}"
            )
            
            # Store group
            self.groups[group_id] = group
            
            # Update user-group mappings
            for member in members:
                if member not in self.user_groups:
                    self.user_groups[member] = set()
                self.user_groups[member].add(group_id)
            
            # Save to disk
            self._save_groups()
            
            logger.info(f"✅ Created group {group_id} with {len(members)} members")
            return group
    
    def get_group(self, group_id: str) -> Optional[Group]:
        """Get a group by ID"""
        return self.groups.get(group_id)
    
    def get_user_groups(self, user_id: str) -> List[Group]:
        """Get all groups a user belongs to"""
        group_ids = self.user_groups.get(user_id, set())
        return [self.groups[gid] for gid in group_ids if gid in self.groups]
    
    def add_member(self, group_id: str, user_id: str) -> bool:
        """Add a member to a group"""
        with self._lock:
            group = self.groups.get(group_id)
            if not group:
                raise ValueError(f"Group {group_id} not found")
            
            # Check if already member
            if user_id in group.members:
                return False
            
            # Check if banned
            if group.is_banned(user_id):
                raise ValueError(f"User {user_id} is banned from group {group_id}")
            
            # Add member
            group.members.append(user_id)
            
            # Update user-group mapping
            if user_id not in self.user_groups:
                self.user_groups[user_id] = set()
            self.user_groups[user_id].add(group_id)
            
            # Save
            self._save_groups()
            
            logger.info(f"➕ Added {user_id} to group {group_id}")
            return True
    
    def remove_member(self, group_id: str, user_id: str) -> bool:
        """Remove a member from a group"""
        with self._lock:
            group = self.groups.get(group_id)
            if not group:
                raise ValueError(f"Group {group_id} not found")
            
            # Check if member
            if user_id not in group.members:
                return False
            
            # Don't allow owner to leave (must transfer ownership first)
            if user_id == group.owner_id:
                raise ValueError("Owner cannot leave group without transferring ownership")
            
            # Remove member
            group.members.remove(user_id)
            
            # Remove from admins if present
            if user_id in group.admins:
                group.admins.remove(user_id)
            
            # Update user-group mapping
            if user_id in self.user_groups:
                self.user_groups[user_id].discard(group_id)
            
            # Save
            self._save_groups()
            
            logger.info(f"➖ Removed {user_id} from group {group_id}")
            return True
    
    def ban_user(self, group_id: str, user_id: str, admin_id: str):
        """Ban a user from a group"""
        with self._lock:
            group = self.groups.get(group_id)
            if not group:
                raise ValueError(f"Group {group_id} not found")
            
            # Check permissions
            if not group.is_admin(admin_id):
                raise ValueError("Only admins can ban users")
            
            # Can't ban owner
            if user_id == group.owner_id:
                raise ValueError("Cannot ban the group owner")
            
            # Add to ban list
            group.banned_users.add(user_id)
            
            # Remove from group if member
            if user_id in group.members:
                self.remove_member(group_id, user_id)
            
            # Save
            self._save_groups()
            
            logger.info(f"🚫 Banned {user_id} from group {group_id}")
    
    def transfer_ownership(self, group_id: str, current_owner_id: str, new_owner_id: str):
        """Transfer group ownership"""
        with self._lock:
            group = self.groups.get(group_id)
            if not group:
                raise ValueError(f"Group {group_id} not found")
            
            # Verify current owner
            if group.owner_id != current_owner_id:
                raise ValueError("Only the current owner can transfer ownership")
            
            # New owner must be a member
            if new_owner_id not in group.members:
                raise ValueError("New owner must be a member of the group")
            
            # Transfer ownership
            group.owner_id = new_owner_id
            
            # Save
            self._save_groups()
            
            logger.info(f"👑 Transferred ownership of {group_id} from {current_owner_id} to {new_owner_id}")
    
    def delete_group(self, group_id: str, owner_id: str):
        """Delete a group (owner only)"""
        with self._lock:
            group = self.groups.get(group_id)
            if not group:
                raise ValueError(f"Group {group_id} not found")
            
            # Verify owner
            if group.owner_id != owner_id:
                raise ValueError("Only the owner can delete the group")
            
            # Remove from all user mappings
            for member in group.members:
                if member in self.user_groups:
                    self.user_groups[member].discard(group_id)
            
            # Delete group
            del self.groups[group_id]
            
            # Save
            self._save_groups()
            
            # TODO: Clean up group's concept mesh data
            
            logger.info(f"🗑️ Deleted group {group_id}")
    
    def _save_groups(self):
        """Save groups to disk"""
        try:
            data = {
                "groups": {gid: g.to_dict() for gid, g in self.groups.items()},
                "user_groups": {uid: list(gids) for uid, gids in self.user_groups.items()},
                "last_updated": time.time()
            }
            
            with open(self.groups_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save groups: {e}")
    
    def _load_groups(self):
        """Load groups from disk"""
        try:
            if self.groups_file.exists():
                with open(self.groups_file, 'r') as f:
                    data = json.load(f)
                
                # Load groups
                for gid, gdata in data.get("groups", {}).items():
                    self.groups[gid] = Group.from_dict(gdata)
                
                # Load user-group mappings
                for uid, gids in data.get("user_groups", {}).items():
                    self.user_groups[uid] = set(gids)
                
                logger.info(f"Loaded {len(self.groups)} groups from disk")
                
        except Exception as e:
            logger.error(f"Failed to load groups: {e}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get group manager statistics"""
        return {
            "total_groups": len(self.groups),
            "total_users": len(self.user_groups),
            "largest_group": max((len(g.members) for g in self.groups.values()), default=0),
            "groups_with_bans": sum(1 for g in self.groups.values() if g.banned_users)
        }
