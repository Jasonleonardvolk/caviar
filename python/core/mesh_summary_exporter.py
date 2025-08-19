#!/usr/bin/env python3
"""
Mesh Context Summary Exporter for TORI
Generates nightly JSON summaries of user's ConceptMesh state for context injection
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import time
from enum import Enum
import threading

# Import schema versioning (Improvement #4)
try:
    from mesh_schema_versioning import (
        get_global_schema_manager,
        add_version_to_mesh,
        write_mesh_safely,
        CURRENT_SCHEMA_VERSION
    )
    VERSIONING_AVAILABLE = True
except ImportError:
    VERSIONING_AVAILABLE = False
    CURRENT_SCHEMA_VERSION = "2.0.0"

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS
# ============================================================================

class ExportTrigger(Enum):
    """Types of events that can trigger mesh export."""
    NIGHTLY = "nightly"           # Scheduled nightly export
    INTENT_CLOSED = "intent_closed"  # Intent was resolved/closed
    DOCUMENT_UPLOAD = "document_upload"  # New document ingested
    CONCEPT_CHANGE = "concept_change"  # Significant concept mesh change
    MANUAL = "manual"             # User-triggered export
    SESSION_END = "session_end"    # High-impact session ended
    
class ExportMode(Enum):
    """Export mode configuration."""
    NIGHTLY = "nightly"    # Only nightly exports
    EVENT = "event"        # Only event-driven exports
    HYBRID = "hybrid"      # Both nightly and event-driven

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ConceptEntry:
    """Represents a concept in the mesh summary."""
    name: str
    summary: str
    score: float = 0.0
    last_active: Optional[str] = None
    source: str = "personal"  # personal/team/global
    keywords: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "name": self.name,
            "summary": self.summary,
            "score": self.score
        }
        if self.last_active:
            result["last_active"] = self.last_active
        if self.keywords:
            result["keywords"] = self.keywords
        return result

@dataclass
class IntentEntry:
    """Represents an unresolved intent."""
    id: str
    description: str
    intent_type: str = "unknown"
    created_at: str = None
    last_active: str = None
    priority: str = "normal"  # low/normal/high/critical
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "intent_type": self.intent_type,
            "last_active": self.last_active or self.created_at,
            "priority": self.priority
        }

@dataclass
class MeshSummary:
    """Complete mesh context summary for a user."""
    user_id: str
    timestamp: str
    personal_concepts: List[ConceptEntry]
    open_intents: List[IntentEntry]
    recent_activity: str
    team_concepts: Dict[str, List[ConceptEntry]]
    global_concepts: List[ConceptEntry]
    groups: List[str] = None  # Group memberships
    schema_version: str = CURRENT_SCHEMA_VERSION  # Schema version
    starred_items: List[str] = None  # Starred item IDs
    
    def to_json(self) -> str:
        """Convert to JSON string with versioning."""
        data = {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "personal_concepts": [c.to_dict() for c in self.personal_concepts],
            "open_intents": [i.to_dict() for i in self.open_intents],
            "recent_activity": self.recent_activity,
            "team_concepts": {
                team: [c.to_dict() for c in concepts]
                for team, concepts in self.team_concepts.items()
            },
            "global_concepts": [c.to_dict() for c in self.global_concepts]
        }
        if self.groups:
            data["groups"] = self.groups
        if self.starred_items:
            data["starred_items"] = self.starred_items
        
        # Add schema metadata
        data["schema_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "generator": "MeshSummaryExporter",
            "version_info": "Full v2.0.0 schema with all features"
        }
        
        return json.dumps(data, indent=2)

# ============================================================================
# MESH SUMMARY EXPORTER
# ============================================================================

class MeshSummaryExporter:
    """
    Exports mesh context summaries for users and groups.
    Supports event-driven, nightly, and hybrid export modes.
    Aggregates data from ConceptMesh, IntentTrace, and MemoryVault.
    """
    
    def __init__(self,
                 mesh_contexts_dir: str = "models/mesh_contexts",
                 memory_vault_dir: str = "memory_vault",
                 concept_mesh_dir: str = "concept_mesh",
                 export_mode: ExportMode = ExportMode.HYBRID,
                 debounce_minutes: int = 10):
        """
        Initialize exporter with event-driven capabilities.
        
        Args:
            mesh_contexts_dir: Where to save summaries
            memory_vault_dir: MemoryVault directory
            concept_mesh_dir: ConceptMesh directory
            export_mode: Export mode (nightly/event/hybrid)
            debounce_minutes: Minimum minutes between exports per user
        """
        self.mesh_contexts_dir = Path(mesh_contexts_dir)
        self.memory_vault_dir = Path(memory_vault_dir)
        self.concept_mesh_dir = Path(concept_mesh_dir)
        self.export_mode = export_mode
        self.debounce_minutes = debounce_minutes
        
        # Create directories if needed
        self.mesh_contexts_dir.mkdir(parents=True, exist_ok=True)
        (self.mesh_contexts_dir / "groups").mkdir(exist_ok=True)
        
        # Event tracking
        self.last_export_time: Dict[str, float] = {}  # user_id -> timestamp
        self.export_lock = threading.Lock()  # Thread safety
        self.export_stats: Dict[str, int] = {trigger.value: 0 for trigger in ExportTrigger}
        
        # Event log file
        self.event_log_path = Path("logs")
        self.event_log_path.mkdir(exist_ok=True)
        self.event_log_file = self.event_log_path / "mesh_export_events.log"
        
        logger.info(f"MeshSummaryExporter initialized: {self.mesh_contexts_dir}")
        logger.info(f"Export mode: {self.export_mode.value}, Debounce: {self.debounce_minutes} min")
    
    def should_export(self, user_id: str, force: bool = False) -> bool:
        """
        Check if export should proceed based on debouncing.
        
        Args:
            user_id: User identifier
            force: Force export regardless of debounce
            
        Returns:
            True if export should proceed
        """
        if force:
            return True
            
        with self.export_lock:
            last_time = self.last_export_time.get(user_id, 0)
            current_time = time.time()
            time_since_last = (current_time - last_time) / 60  # Convert to minutes
            
            if time_since_last >= self.debounce_minutes:
                return True
            else:
                logger.debug(f"Debounce active for {user_id}: {self.debounce_minutes - time_since_last:.1f} min remaining")
                return False
    
    def log_export_event(self, user_id: str, trigger: ExportTrigger, duration: float, success: bool):
        """
        Log export event to event log file.
        
        Args:
            user_id: User identifier
            trigger: What triggered the export
            duration: Export duration in seconds
            success: Whether export succeeded
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "user_id": user_id,
            "trigger": trigger.value,
            "duration_seconds": round(duration, 2),
            "success": success,
            "mode": self.export_mode.value
        }
        
        try:
            with open(self.event_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log export event: {e}")
        
        # Update statistics
        if success:
            self.export_stats[trigger.value] += 1
    
    def trigger_export(self, 
                      user_id: str, 
                      trigger: ExportTrigger,
                      force: bool = False,
                      metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Trigger mesh export with event tracking.
        
        Args:
            user_id: User identifier
            trigger: What triggered this export
            force: Force export regardless of debounce
            metadata: Additional metadata about the trigger
            
        Returns:
            Path to exported summary or None if skipped
        """
        # Check mode compatibility
        if self.export_mode == ExportMode.NIGHTLY and trigger != ExportTrigger.NIGHTLY:
            logger.debug(f"Skipping {trigger.value} export in NIGHTLY mode")
            return None
        
        if self.export_mode == ExportMode.EVENT and trigger == ExportTrigger.NIGHTLY:
            logger.debug("Skipping nightly export in EVENT mode")
            return None
        
        # Check debounce
        if not self.should_export(user_id, force):
            return None
        
        # Perform export
        start_time = time.time()
        success = False
        result_path = None
        
        try:
            logger.info(f"Triggered export for {user_id} due to {trigger.value}")
            if metadata:
                logger.debug(f"Trigger metadata: {metadata}")
            
            result_path = self.export_user_mesh_summary(user_id)
            success = True
            
            # Update last export time
            with self.export_lock:
                self.last_export_time[user_id] = time.time()
                
        except Exception as e:
            logger.error(f"Export failed for {user_id}: {e}")
        
        # Log event
        duration = time.time() - start_time
        self.log_export_event(user_id, trigger, duration, success)
        
        if success:
            logger.info(f"Export completed in {duration:.2f}s: {result_path}")
        
        return result_path
    
    def export_user_mesh_summary(self, user_id: str) -> str:
        """
        Generate and export mesh summary for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Path to saved summary file
        """
        logger.info(f"Exporting mesh summary for user: {user_id}")
        
        # Gather data
        personal_concepts = self._get_top_concepts(user_id, scope="personal", top_n=10)
        open_intents = self._get_open_intents(user_id)
        recent_activity = self._get_recent_activity(user_id)
        team_concepts = self._get_team_concepts(user_id)
        global_concepts = self._get_relevant_global_concepts(user_id, top_n=3)
        groups = self._get_user_groups(user_id)
        
        # Create summary
        summary = MeshSummary(
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            personal_concepts=personal_concepts,
            open_intents=open_intents,
            recent_activity=recent_activity,
            team_concepts=team_concepts,
            global_concepts=global_concepts,
            groups=groups
        )
        
        # Save to file
        output_path = self.mesh_contexts_dir / f"{user_id}_mesh.json"
        with open(output_path, 'w') as f:
            f.write(summary.to_json())
        
        logger.info(f"Exported summary to {output_path}")
        logger.info(f"  - {len(personal_concepts)} personal concepts")
        logger.info(f"  - {len(open_intents)} open intents")
        logger.info(f"  - {sum(len(c) for c in team_concepts.values())} team concepts")
        
        return str(output_path)
    
    def export_group_mesh_summary(self, group_id: str) -> str:
        """
        Generate and export mesh summary for a group/team.
        
        Args:
            group_id: Group identifier
            
        Returns:
            Path to saved summary file
        """
        logger.info(f"Exporting mesh summary for group: {group_id}")
        
        # Gather group data
        group_concepts = self._get_group_concepts(group_id, top_n=20)
        group_intents = self._get_group_intents(group_id)
        group_activity = self._get_group_activity(group_id)
        
        # Create summary (simplified structure for groups)
        summary = {
            "group_id": group_id,
            "timestamp": datetime.now().isoformat(),
            "concepts": [c.to_dict() for c in group_concepts],
            "shared_intents": [i.to_dict() for i in group_intents],
            "recent_activity": group_activity,
            "members": self._get_group_members(group_id)
        }
        
        # Save to file
        output_path = self.mesh_contexts_dir / "groups" / f"{group_id}_mesh.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Exported group summary to {output_path}")
        return str(output_path)
    
    def _get_top_concepts(self, 
                         user_id: str, 
                         scope: str = "personal",
                         top_n: int = 10) -> List[ConceptEntry]:
        """
        Get top concepts from ConceptMesh.
        
        Args:
            user_id: User identifier
            scope: Concept scope (personal/team/global)
            top_n: Number of top concepts
            
        Returns:
            List of top concept entries
        """
        concepts = []
        
        # Try to load from ConceptMesh files
        mesh_file = self.concept_mesh_dir / f"{user_id}_mesh.json"
        if mesh_file.exists():
            try:
                with open(mesh_file, 'r') as f:
                    mesh_data = json.load(f)
                
                # Extract nodes
                nodes = mesh_data.get("nodes", {})
                
                # Filter by scope and sort by relevance/frequency
                scope_nodes = []
                for node_id, node_data in nodes.items():
                    if node_data.get("scope", "personal") == scope:
                        # Calculate score based on connections and activity
                        score = len(node_data.get("connections", [])) * 0.5
                        score += node_data.get("frequency", 0) * 0.3
                        score += node_data.get("importance", 0) * 0.2
                        
                        scope_nodes.append((node_data, score))
                
                # Sort by score and take top N
                scope_nodes.sort(key=lambda x: x[1], reverse=True)
                
                for node_data, score in scope_nodes[:top_n]:
                    concept = ConceptEntry(
                        name=node_data.get("label", "Unknown"),
                        summary=node_data.get("description", ""),
                        score=min(score / 10, 1.0),  # Normalize to 0-1
                        source=scope,
                        keywords=node_data.get("keywords", [])
                    )
                    concepts.append(concept)
            
            except Exception as e:
                logger.error(f"Error loading concept mesh: {e}")
        
        # Fallback: Generate some default concepts based on recent activity
        if not concepts:
            # Analyze recent conversations for key topics
            recent_topics = self._extract_topics_from_memory(user_id)
            for i, topic in enumerate(recent_topics[:top_n]):
                concepts.append(ConceptEntry(
                    name=topic,
                    summary=f"Recent topic of interest",
                    score=0.5 - (i * 0.05),
                    source=scope
                ))
        
        return concepts
    
    def _get_open_intents(self, user_id: str) -> List[IntentEntry]:
        """
        Get unresolved intents from IntentTrace.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of open intent entries
        """
        intents = []
        
        # Check for open intents file
        intents_file = self.memory_vault_dir / "intents" / f"{user_id}_open_intents.json"
        if intents_file.exists():
            try:
                with open(intents_file, 'r') as f:
                    intent_data = json.load(f)
                
                for intent in intent_data.get("open_intents", []):
                    entry = IntentEntry(
                        id=intent.get("id", f"intent_{len(intents)}"),
                        description=intent.get("description", ""),
                        intent_type=intent.get("type", "unknown"),
                        created_at=intent.get("created_at"),
                        last_active=intent.get("last_active"),
                        priority=intent.get("priority", "normal")
                    )
                    intents.append(entry)
            
            except Exception as e:
                logger.error(f"Error loading open intents: {e}")
        
        # Also check trace files for OPEN intents
        traces_dir = self.memory_vault_dir / "traces"
        if traces_dir.exists():
            for trace_file in traces_dir.glob(f"*{user_id}*.jsonl"):
                try:
                    with open(trace_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                event = json.loads(line)
                                if (event.get("status") == "OPEN" and 
                                    event.get("event") == "intent_opened"):
                                    # Add if not already in list
                                    intent_id = event.get("intent_id")
                                    if not any(i.id == intent_id for i in intents):
                                        intents.append(IntentEntry(
                                            id=intent_id,
                                            description=event.get("description", ""),
                                            intent_type=event.get("intent_type", "unknown"),
                                            created_at=event.get("timestamp"),
                                            last_active=event.get("timestamp")
                                        ))
                except Exception as e:
                    logger.debug(f"Error reading trace file {trace_file}: {e}")
        
        # Limit to most recent/important
        return intents[:10]
    
    def _get_recent_activity(self, user_id: str, window_days: int = 1) -> str:
        """
        Generate summary of recent user activity.
        
        Args:
            user_id: User identifier
            window_days: Days to look back
            
        Returns:
            Activity summary string
        """
        activities = []
        cutoff_time = datetime.now() - timedelta(days=window_days)
        
        # Check recent session files
        sessions_dir = self.memory_vault_dir / "sessions"
        if sessions_dir.exists():
            for session_file in sessions_dir.glob(f"*{user_id}*.jsonl"):
                try:
                    # Get file modification time
                    file_time = datetime.fromtimestamp(session_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        continue
                    
                    # Extract key topics from recent sessions
                    with open(session_file, 'r') as f:
                        topics = set()
                        for line in f:
                            if line.strip():
                                event = json.loads(line)
                                if event.get("event") == "conversation":
                                    text = event.get("text", "")
                                    # Simple keyword extraction (could be more sophisticated)
                                    words = text.lower().split()
                                    for word in words:
                                        if len(word) > 5 and word.isalpha():
                                            topics.add(word)
                        
                        if topics:
                            activities.append(f"Discussed: {', '.join(list(topics)[:5])}")
                
                except Exception as e:
                    logger.debug(f"Error reading session file: {e}")
        
        # Generate summary
        if activities:
            return " | ".join(activities[:3])
        else:
            return "No recent activity in the last day"
    
    def _get_team_concepts(self, user_id: str) -> Dict[str, List[ConceptEntry]]:
        """
        Get team/shared concepts for user's groups.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of team -> concepts
        """
        team_concepts = {}
        
        # Get user's groups
        groups = self._get_user_groups(user_id)
        
        for group in groups:
            # Load group summary if exists
            group_file = self.mesh_contexts_dir / "groups" / f"{group}_mesh.json"
            if group_file.exists():
                try:
                    with open(group_file, 'r') as f:
                        group_data = json.load(f)
                    
                    concepts = []
                    for concept_dict in group_data.get("concepts", [])[:5]:
                        concepts.append(ConceptEntry(
                            name=concept_dict.get("name"),
                            summary=concept_dict.get("summary"),
                            score=concept_dict.get("score", 0.5),
                            source="team"
                        ))
                    
                    if concepts:
                        team_concepts[group] = concepts
                
                except Exception as e:
                    logger.error(f"Error loading group summary: {e}")
        
        return team_concepts
    
    def _get_relevant_global_concepts(self, user_id: str, top_n: int = 3) -> List[ConceptEntry]:
        """
        Get globally relevant concepts.
        
        Args:
            user_id: User identifier
            top_n: Number of concepts
            
        Returns:
            List of global concepts
        """
        # For now, return empty list as global concepts are usually in base model
        # Could be extended to include important global updates
        return []
    
    def _get_user_groups(self, user_id: str) -> List[str]:
        """
        Get groups the user belongs to.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of group IDs
        """
        groups = []
        
        # Check user profile or config
        profile_file = Path("models/adapters") / f"{user_id}_profile.json"
        if profile_file.exists():
            try:
                with open(profile_file, 'r') as f:
                    profile = json.load(f)
                    groups = profile.get("groups", [])
            except Exception as e:
                logger.debug(f"Error loading user profile: {e}")
        
        # Default groups based on patterns
        if not groups and user_id in ["jason", "alice", "bob"]:
            groups = ["ProjectX"]  # Default test group
        
        return groups
    
    def _get_group_concepts(self, group_id: str, top_n: int = 20) -> List[ConceptEntry]:
        """
        Get concepts for a group.
        
        Args:
            group_id: Group identifier
            top_n: Number of concepts
            
        Returns:
            List of group concepts
        """
        concepts = []
        
        # Load from shared documents or team knowledge base
        # This is a placeholder - would integrate with real document store
        team_docs_dir = Path("data/team_documents") / group_id
        if team_docs_dir.exists():
            # Extract concepts from team documents
            for doc_file in team_docs_dir.glob("*.txt"):
                try:
                    with open(doc_file, 'r') as f:
                        content = f.read()
                        # Simple concept extraction (could use NLP)
                        doc_name = doc_file.stem
                        concepts.append(ConceptEntry(
                            name=doc_name,
                            summary=content[:100],
                            score=0.7,
                            source="team"
                        ))
                except Exception as e:
                    logger.debug(f"Error reading team doc: {e}")
        
        # Add some default team concepts for testing
        if not concepts:
            concepts = [
                ConceptEntry("Beta Algorithm", "Shared algorithm from research paper", 0.8, source="team"),
                ConceptEntry("Q4 Planning", "Quarter 4 project planning", 0.7, source="team"),
                ConceptEntry("Team Standards", "Coding and documentation standards", 0.6, source="team")
            ]
        
        return concepts[:top_n]
    
    def _get_group_intents(self, group_id: str) -> List[IntentEntry]:
        """Get shared intents for a group."""
        # Placeholder - would aggregate from team members
        return []
    
    def _get_group_activity(self, group_id: str) -> str:
        """Get recent group activity summary."""
        return f"Group {group_id} recent collaborative work"
    
    def _get_group_members(self, group_id: str) -> List[str]:
        """Get members of a group."""
        # Placeholder - would come from group management system
        if group_id == "ProjectX":
            return ["alice", "bob", "jason"]
        return []
    
    def _extract_topics_from_memory(self, user_id: str) -> List[str]:
        """
        Extract key topics from recent memory/conversations.
        Simple keyword extraction as fallback.
        """
        topics = []
        word_freq = {}
        
        # Scan recent conversations
        sessions_dir = self.memory_vault_dir / "sessions"
        if sessions_dir.exists():
            for session_file in sessions_dir.glob(f"*{user_id}*.jsonl"):
                try:
                    with open(session_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                event = json.loads(line)
                                if event.get("event") == "conversation":
                                    text = event.get("text", "").lower()
                                    # Count word frequency
                                    words = text.split()
                                    for word in words:
                                        if len(word) > 4 and word.isalpha():
                                            word_freq[word] = word_freq.get(word, 0) + 1
                except Exception:
                    pass
        
        # Get top words as topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        topics = [word for word, _ in sorted_words[:10]]
        
        return topics
    
    def export_all_users(self, user_ids: List[str]) -> Dict[str, str]:
        """
        Export summaries for multiple users.
        
        Args:
            user_ids: List of user IDs
            
        Returns:
            Dictionary of user_id -> file path
        """
        results = {}
        for user_id in user_ids:
            try:
                path = self.export_user_mesh_summary(user_id)
                results[user_id] = path
            except Exception as e:
                logger.error(f"Failed to export for {user_id}: {e}")
                results[user_id] = None
        
        return results

    def get_export_statistics(self) -> Dict[str, Any]:
        """
        Get export statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "mode": self.export_mode.value,
            "debounce_minutes": self.debounce_minutes,
            "trigger_counts": self.export_stats.copy(),
            "users_with_recent_exports": list(self.last_export_time.keys()),
            "total_exports": sum(self.export_stats.values())
        }
    
    def bulk_trigger_export(self,
                           user_ids: List[str],
                           trigger: ExportTrigger,
                           force: bool = False) -> Dict[str, Optional[str]]:
        """
        Trigger export for multiple users.
        
        Args:
            user_ids: List of user IDs
            trigger: What triggered these exports
            force: Force export regardless of debounce
            
        Returns:
            Dictionary of user_id -> export path (or None)
        """
        results = {}
        for user_id in user_ids:
            results[user_id] = self.trigger_export(user_id, trigger, force)
        return results

# ============================================================================
# GLOBAL INSTANCE (Singleton Pattern)
# ============================================================================

_global_exporter: Optional[MeshSummaryExporter] = None

def get_global_exporter(config: Optional[Dict[str, Any]] = None) -> MeshSummaryExporter:
    """
    Get or create global exporter instance.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Global MeshSummaryExporter instance
    """
    global _global_exporter
    
    if _global_exporter is None:
        if config:
            _global_exporter = MeshSummaryExporter(
                export_mode=ExportMode(config.get("export_mode", "hybrid")),
                debounce_minutes=config.get("debounce_minutes", 10)
            )
        else:
            _global_exporter = MeshSummaryExporter()
    
    return _global_exporter

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_nightly_export(user_ids: Optional[List[str]] = None):
    """
    Run nightly export for all users.
    
    Args:
        user_ids: List of users (None = all active users)
    """
    exporter = get_global_exporter()
    
    # Default to known users if not specified
    if not user_ids:
        user_ids = ["jason", "alice", "bob"]  # Would come from user registry
    
    logger.info(f"Starting nightly mesh export for {len(user_ids)} users")
    
    # Use bulk trigger with NIGHTLY trigger
    results = exporter.bulk_trigger_export(user_ids, ExportTrigger.NIGHTLY, force=True)
    
    # Export group summaries
    groups = set()
    for user_id in user_ids:
        user_groups = exporter._get_user_groups(user_id)
        groups.update(user_groups)
    
    for group_id in groups:
        try:
            exporter.export_group_mesh_summary(group_id)
            logger.info(f"Exported group summary for {group_id}")
        except Exception as e:
            logger.error(f"Failed to export group {group_id}: {e}")
    
    logger.info(f"Nightly export complete: {len(results)} users, {len(groups)} groups")
    return results

def trigger_intent_closed_export(user_id: str, intent_id: str, intent_type: str = "unknown"):
    """
    Trigger export after intent closure.
    
    Args:
        user_id: User who closed the intent
        intent_id: Intent identifier
        intent_type: Type of intent
    """
    exporter = get_global_exporter()
    metadata = {
        "intent_id": intent_id,
        "intent_type": intent_type,
        "closed_at": datetime.now().isoformat()
    }
    return exporter.trigger_export(user_id, ExportTrigger.INTENT_CLOSED, metadata=metadata)

def trigger_document_upload_export(user_id: str, document_name: str, document_type: str = "unknown"):
    """
    Trigger export after document upload.
    
    Args:
        user_id: User who uploaded
        document_name: Name of document
        document_type: Type of document
    """
    exporter = get_global_exporter()
    metadata = {
        "document_name": document_name,
        "document_type": document_type,
        "uploaded_at": datetime.now().isoformat()
    }
    return exporter.trigger_export(user_id, ExportTrigger.DOCUMENT_UPLOAD, metadata=metadata)

def trigger_concept_change_export(user_id: str, change_type: str, concept_count: int):
    """
    Trigger export after significant concept mesh change.
    
    Args:
        user_id: User whose mesh changed
        change_type: Type of change (add/remove/merge)
        concept_count: Number of concepts affected
    """
    # Only trigger if change is significant (e.g., > 5 concepts)
    if concept_count < 5:
        logger.debug(f"Concept change too small ({concept_count}), skipping export")
        return None
    
    exporter = get_global_exporter()
    metadata = {
        "change_type": change_type,
        "concept_count": concept_count,
        "changed_at": datetime.now().isoformat()
    }
    return exporter.trigger_export(user_id, ExportTrigger.CONCEPT_CHANGE, metadata=metadata)

def trigger_manual_export(user_id: str, reason: str = "User requested"):
    """
    Trigger manual export by user request.
    
    Args:
        user_id: User requesting export
        reason: Reason for manual export
    """
    exporter = get_global_exporter()
    metadata = {
        "reason": reason,
        "requested_at": datetime.now().isoformat()
    }
    return exporter.trigger_export(user_id, ExportTrigger.MANUAL, force=True, metadata=metadata)

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "ConceptEntry",
    "IntentEntry",
    "MeshSummary",
    "MeshSummaryExporter",
    "ExportTrigger",
    "ExportMode",
    "get_global_exporter",
    "run_nightly_export",
    "trigger_intent_closed_export",
    "trigger_document_upload_export",
    "trigger_concept_change_export",
    "trigger_manual_export"
]

# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Test export
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
        exporter = MeshSummaryExporter()
        path = exporter.export_user_mesh_summary(user_id)
        print(f"Exported summary to: {path}")
        
        # Show content
        with open(path, 'r') as f:
            print("\nSummary content:")
            print(f.read())
    else:
        # Run nightly export for all
        results = run_nightly_export()
        print("\nExport results:")
        for user_id, path in results.items():
            print(f"  {user_id}: {path}")
