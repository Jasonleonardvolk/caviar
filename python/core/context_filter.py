#!/usr/bin/env python3
"""
Context Weighting & Query-Relevance Filtering for TORI
Intelligently selects and weights mesh context based on prompt relevance
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re
import hashlib

# Optional: For embedding-based similarity
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Embedding similarity disabled.")

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class WeightingMode(Enum):
    """Context weighting strategies."""
    NONE = "none"              # No filtering (original behavior)
    KEYWORD = "keyword"        # Keyword matching only
    EMBEDDING = "embedding"    # Embedding similarity only
    HYBRID = "hybrid"          # Combined keyword + embedding
    SMART = "smart"           # Adaptive based on query type

class ContextType(Enum):
    """Types of context elements."""
    PERSONAL_CONCEPT = "personal_concept"
    TEAM_CONCEPT = "team_concept"
    GLOBAL_CONCEPT = "global_concept"
    OPEN_INTENT = "open_intent"
    RECENT_ACTIVITY = "recent_activity"
    GROUP_CONTEXT = "group_context"

# Default weights for different factors
DEFAULT_WEIGHTS = {
    "keyword_match": 0.4,
    "embedding_similarity": 0.3,
    "recency": 0.15,
    "priority": 0.1,
    "user_weight": 0.05  # For starred/pinned items
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ScoredContextItem:
    """A context item with relevance score."""
    content: Dict[str, Any]
    context_type: ContextType
    score: float
    reasons: List[str] = field(default_factory=list)
    user_starred: bool = False
    user_weight: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "type": self.context_type.value,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "starred": self.user_starred
        }

@dataclass
class FilterConfig:
    """Configuration for context filtering."""
    mode: WeightingMode = WeightingMode.HYBRID
    max_personal_concepts: int = 5
    max_team_concepts: int = 3
    max_open_intents: int = 3
    max_total_items: int = 10
    max_context_tokens: int = 200
    
    # Score thresholds
    min_relevance_score: float = 0.1
    star_boost_factor: float = 2.0  # Multiplier for starred items
    
    # Component weights
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    
    # Embedding config
    embedding_model: str = "all-MiniLM-L6-v2"
    cache_embeddings: bool = True
    embedding_cache_dir: str = "models/embeddings_cache"

# ============================================================================
# CONTEXT FILTER
# ============================================================================

class ContextFilter:
    """
    Filters and weights mesh context based on query relevance.
    Supports keyword matching, embedding similarity, and user preferences.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize context filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config or FilterConfig()
        
        # Initialize embedding model if available and needed
        self.embedding_model = None
        if EMBEDDINGS_AVAILABLE and self.config.mode in [WeightingMode.EMBEDDING, WeightingMode.HYBRID, WeightingMode.SMART]:
            try:
                self.embedding_model = SentenceTransformer(self.config.embedding_model)
                logger.info(f"Loaded embedding model: {self.config.embedding_model}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                # Fallback to keyword mode
                if self.config.mode == WeightingMode.EMBEDDING:
                    self.config.mode = WeightingMode.KEYWORD
                    logger.warning("Falling back to keyword mode")
        
        # Embedding cache
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_dir = Path(self.config.embedding_cache_dir)
        if self.config.cache_embeddings:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_embedding_cache()
        
        # User preferences (starred/pinned items)
        self.user_preferences: Dict[str, Dict[str, float]] = {}
        self._load_user_preferences()
        
        logger.info(f"ContextFilter initialized with mode: {self.config.mode.value}")
    
    def filter_relevant_context(self,
                               mesh_context: Dict[str, Any],
                               prompt: str,
                               user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Filter mesh context to only include relevant items for the prompt.
        
        Args:
            mesh_context: Full mesh context dictionary
            prompt: User's prompt/query
            user_id: User identifier for preferences
            
        Returns:
            Filtered mesh context with only relevant items
        """
        if self.config.mode == WeightingMode.NONE:
            return mesh_context
        
        # Score all context items
        scored_items = self._score_all_items(mesh_context, prompt, user_id)
        
        # Select top items per category
        selected = self._select_top_items(scored_items)
        
        # Reconstruct filtered context
        filtered_context = self._reconstruct_context(selected, mesh_context)
        
        # Log filtering statistics
        self._log_filtering_stats(mesh_context, filtered_context, selected)
        
        return filtered_context
    
    def _score_all_items(self,
                        mesh_context: Dict[str, Any],
                        prompt: str,
                        user_id: Optional[str] = None) -> List[ScoredContextItem]:
        """
        Score all context items for relevance.
        
        Args:
            mesh_context: Full mesh context
            prompt: User's prompt
            user_id: User identifier
            
        Returns:
            List of scored items
        """
        scored_items = []
        prompt_lower = prompt.lower()
        prompt_keywords = self._extract_keywords(prompt)
        
        # Get prompt embedding if available
        prompt_embedding = None
        if self.embedding_model and self.config.mode in [WeightingMode.EMBEDDING, WeightingMode.HYBRID, WeightingMode.SMART]:
            prompt_embedding = self._get_embedding(prompt)
        
        # Score personal concepts
        if "personal_concepts" in mesh_context:
            for concept in mesh_context["personal_concepts"]:
                score, reasons = self._score_concept(
                    concept, prompt_lower, prompt_keywords, prompt_embedding,
                    ContextType.PERSONAL_CONCEPT, user_id
                )
                if score >= self.config.min_relevance_score:
                    scored_items.append(ScoredContextItem(
                        content=concept,
                        context_type=ContextType.PERSONAL_CONCEPT,
                        score=score,
                        reasons=reasons,
                        user_starred=self._is_starred(user_id, concept.get("name")),
                        user_weight=self._get_user_weight(user_id, concept.get("name"))
                    ))
        
        # Score open intents
        if "open_intents" in mesh_context:
            for intent in mesh_context["open_intents"]:
                score, reasons = self._score_intent(
                    intent, prompt_lower, prompt_keywords, prompt_embedding, user_id
                )
                if score >= self.config.min_relevance_score:
                    scored_items.append(ScoredContextItem(
                        content=intent,
                        context_type=ContextType.OPEN_INTENT,
                        score=score,
                        reasons=reasons,
                        user_starred=self._is_starred(user_id, intent.get("id")),
                        user_weight=self._get_user_weight(user_id, intent.get("id"))
                    ))
        
        # Score team concepts
        if "team_concepts" in mesh_context:
            for team, concepts in mesh_context["team_concepts"].items():
                for concept in concepts:
                    score, reasons = self._score_concept(
                        concept, prompt_lower, prompt_keywords, prompt_embedding,
                        ContextType.TEAM_CONCEPT, user_id, team_context=team
                    )
                    if score >= self.config.min_relevance_score:
                        concept["team"] = team  # Add team info
                        scored_items.append(ScoredContextItem(
                            content=concept,
                            context_type=ContextType.TEAM_CONCEPT,
                            score=score,
                            reasons=reasons,
                            user_starred=self._is_starred(user_id, f"{team}:{concept.get('name')}"),
                            user_weight=self._get_user_weight(user_id, f"{team}:{concept.get('name')}")
                        ))
        
        # Score recent activity
        if "recent_activity" in mesh_context:
            activity = mesh_context["recent_activity"]
            if activity and activity != "No recent activity in the last day":
                score, reasons = self._score_activity(
                    activity, prompt_lower, prompt_keywords, prompt_embedding
                )
                if score >= self.config.min_relevance_score:
                    scored_items.append(ScoredContextItem(
                        content={"activity": activity},
                        context_type=ContextType.RECENT_ACTIVITY,
                        score=score,
                        reasons=reasons
                    ))
        
        # Sort by score
        scored_items.sort(key=lambda x: x.score, reverse=True)
        
        return scored_items
    
    def _score_concept(self,
                      concept: Dict[str, Any],
                      prompt_lower: str,
                      prompt_keywords: Set[str],
                      prompt_embedding: Optional[np.ndarray],
                      context_type: ContextType,
                      user_id: Optional[str] = None,
                      team_context: Optional[str] = None) -> Tuple[float, List[str]]:
        """
        Score a concept for relevance.
        
        Returns:
            (score, list of scoring reasons)
        """
        score = 0.0
        reasons = []
        
        concept_name = concept.get("name", "").lower()
        concept_summary = concept.get("summary", "").lower()
        concept_keywords = set(kw.lower() for kw in concept.get("keywords", []))
        
        # 1. Keyword matching
        keyword_score = 0.0
        
        # Direct name match
        if concept_name in prompt_lower:
            keyword_score += 1.0
            reasons.append(f"name match: '{concept.get('name')}'")
        
        # Keyword overlap
        prompt_kw_set = set(prompt_keywords)
        concept_kw_set = concept_keywords | {concept_name}
        overlap = prompt_kw_set & concept_kw_set
        if overlap:
            keyword_score += 0.5 * len(overlap) / max(len(prompt_kw_set), 1)
            reasons.append(f"keyword overlap: {overlap}")
        
        # Summary contains prompt keywords
        for kw in prompt_keywords:
            if kw in concept_summary:
                keyword_score += 0.2
                reasons.append(f"'{kw}' in summary")
        
        keyword_score = min(keyword_score, 1.0)  # Cap at 1.0
        score += keyword_score * self.config.weights["keyword_match"]
        
        # 2. Embedding similarity
        if self.embedding_model and prompt_embedding is not None:
            concept_text = f"{concept.get('name', '')} {concept.get('summary', '')}"
            concept_embedding = self._get_embedding(concept_text, cache_key=concept.get("name"))
            
            if concept_embedding is not None:
                similarity = self._cosine_similarity(prompt_embedding, concept_embedding)
                score += similarity * self.config.weights["embedding_similarity"]
                if similarity > 0.5:
                    reasons.append(f"high similarity: {similarity:.2f}")
        
        # 3. Recency weighting
        concept_score = concept.get("score", 0.5)  # Concept's own score (0-1)
        score += concept_score * self.config.weights["recency"]
        if concept_score > 0.7:
            reasons.append(f"high activity: {concept_score:.2f}")
        
        # 4. Priority (for team concepts)
        if context_type == ContextType.TEAM_CONCEPT:
            score *= 0.8  # Slightly lower weight for team concepts
        
        # 5. User weight (starred/pinned)
        if user_id:
            item_key = f"{team_context}:{concept.get('name')}" if team_context else concept.get("name")
            user_weight = self._get_user_weight(user_id, item_key)
            if user_weight > 0:
                score += user_weight * self.config.weights["user_weight"]
                if self._is_starred(user_id, item_key):
                    score *= self.config.star_boost_factor
                    reasons.append("starred by user")
        
        return score, reasons
    
    def _score_intent(self,
                     intent: Dict[str, Any],
                     prompt_lower: str,
                     prompt_keywords: Set[str],
                     prompt_embedding: Optional[np.ndarray],
                     user_id: Optional[str] = None) -> Tuple[float, List[str]]:
        """
        Score an intent for relevance.
        
        Returns:
            (score, list of scoring reasons)
        """
        score = 0.0
        reasons = []
        
        intent_desc = intent.get("description", "").lower()
        intent_type = intent.get("intent_type", "").lower()
        
        # 1. Keyword matching
        keyword_score = 0.0
        
        # Check if prompt references the intent
        for kw in prompt_keywords:
            if kw in intent_desc:
                keyword_score += 0.3
                reasons.append(f"'{kw}' in intent")
        
        # Check for question words that match open intents
        question_words = {"how", "what", "why", "when", "where", "can", "should"}
        if any(qw in prompt_lower for qw in question_words):
            keyword_score += 0.2
            reasons.append("question matches open intent")
        
        keyword_score = min(keyword_score, 1.0)
        score += keyword_score * self.config.weights["keyword_match"]
        
        # 2. Embedding similarity
        if self.embedding_model and prompt_embedding is not None:
            intent_embedding = self._get_embedding(intent_desc, cache_key=intent.get("id"))
            
            if intent_embedding is not None:
                similarity = self._cosine_similarity(prompt_embedding, intent_embedding)
                score += similarity * self.config.weights["embedding_similarity"]
                if similarity > 0.6:
                    reasons.append(f"related intent: {similarity:.2f}")
        
        # 3. Priority weighting
        priority_scores = {"critical": 1.0, "high": 0.7, "normal": 0.4, "low": 0.2}
        priority = intent.get("priority", "normal")
        priority_score = priority_scores.get(priority, 0.4)
        score += priority_score * self.config.weights["priority"]
        if priority in ["critical", "high"]:
            reasons.append(f"{priority} priority")
        
        # 4. Recency
        last_active = intent.get("last_active")
        if last_active:
            try:
                last_active_dt = datetime.fromisoformat(last_active.replace("Z", "+00:00"))
                days_old = (datetime.now(last_active_dt.tzinfo) - last_active_dt).days
                if days_old <= 1:
                    score += 0.5 * self.config.weights["recency"]
                    reasons.append("recently active")
                elif days_old <= 7:
                    score += 0.2 * self.config.weights["recency"]
            except:
                pass
        
        # 5. User weight
        if user_id:
            user_weight = self._get_user_weight(user_id, intent.get("id"))
            if user_weight > 0:
                score += user_weight * self.config.weights["user_weight"]
                if self._is_starred(user_id, intent.get("id")):
                    score *= self.config.star_boost_factor
                    reasons.append("starred intent")
        
        return score, reasons
    
    def _score_activity(self,
                       activity: str,
                       prompt_lower: str,
                       prompt_keywords: Set[str],
                       prompt_embedding: Optional[np.ndarray]) -> Tuple[float, List[str]]:
        """
        Score recent activity for relevance.
        
        Returns:
            (score, list of scoring reasons)
        """
        score = 0.0
        reasons = []
        activity_lower = activity.lower()
        
        # Keyword matching
        keyword_score = 0.0
        for kw in prompt_keywords:
            if kw in activity_lower:
                keyword_score += 0.3
                reasons.append(f"'{kw}' in recent activity")
        
        keyword_score = min(keyword_score, 1.0)
        score += keyword_score * self.config.weights["keyword_match"]
        
        # Embedding similarity
        if self.embedding_model and prompt_embedding is not None:
            activity_embedding = self._get_embedding(activity)
            if activity_embedding is not None:
                similarity = self._cosine_similarity(prompt_embedding, activity_embedding)
                score += similarity * self.config.weights["embedding_similarity"]
                if similarity > 0.4:
                    reasons.append(f"related activity: {similarity:.2f}")
        
        # Recency boost (always recent by definition)
        score += 0.5 * self.config.weights["recency"]
        
        return score, reasons
    
    def _select_top_items(self, scored_items: List[ScoredContextItem]) -> List[ScoredContextItem]:
        """
        Select top items per category within limits.
        
        Args:
            scored_items: All scored items
            
        Returns:
            Selected items
        """
        selected = []
        counts = {ct: 0 for ct in ContextType}
        
        # First pass: Add all starred items (they bypass limits)
        for item in scored_items:
            if item.user_starred:
                selected.append(item)
                counts[item.context_type] += 1
        
        # Second pass: Add non-starred items up to limits
        for item in scored_items:
            if item in selected:
                continue
            
            # Check category limit
            if item.context_type == ContextType.PERSONAL_CONCEPT:
                if counts[ContextType.PERSONAL_CONCEPT] >= self.config.max_personal_concepts:
                    continue
            elif item.context_type == ContextType.TEAM_CONCEPT:
                if counts[ContextType.TEAM_CONCEPT] >= self.config.max_team_concepts:
                    continue
            elif item.context_type == ContextType.OPEN_INTENT:
                if counts[ContextType.OPEN_INTENT] >= self.config.max_open_intents:
                    continue
            
            # Check total limit
            if len(selected) >= self.config.max_total_items:
                break
            
            selected.append(item)
            counts[item.context_type] += 1
        
        return selected
    
    def _reconstruct_context(self,
                           selected: List[ScoredContextItem],
                           original: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct filtered context from selected items.
        
        Args:
            selected: Selected items
            original: Original context
            
        Returns:
            Filtered context
        """
        filtered = {
            "user_id": original.get("user_id"),
            "timestamp": original.get("timestamp"),
            "filtering_applied": True,
            "filter_mode": self.config.mode.value
        }
        
        # Group selected items by type
        personal_concepts = []
        team_concepts = {}
        open_intents = []
        recent_activity = None
        
        for item in selected:
            if item.context_type == ContextType.PERSONAL_CONCEPT:
                personal_concepts.append(item.content)
            elif item.context_type == ContextType.TEAM_CONCEPT:
                team = item.content.pop("team", "unknown")
                if team not in team_concepts:
                    team_concepts[team] = []
                team_concepts[team].append(item.content)
            elif item.context_type == ContextType.OPEN_INTENT:
                open_intents.append(item.content)
            elif item.context_type == ContextType.RECENT_ACTIVITY:
                recent_activity = item.content.get("activity")
        
        # Add to filtered context
        if personal_concepts:
            filtered["personal_concepts"] = personal_concepts
        if team_concepts:
            filtered["team_concepts"] = team_concepts
        if open_intents:
            filtered["open_intents"] = open_intents
        if recent_activity:
            filtered["recent_activity"] = recent_activity
        
        # Preserve groups if present
        if "groups" in original:
            filtered["groups"] = original["groups"]
        
        # Add filtering metadata
        filtered["filter_stats"] = {
            "original_items": self._count_items(original),
            "filtered_items": self._count_items(filtered),
            "top_scores": [item.score for item in selected[:5]]
        }
        
        return filtered
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Remove common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "shall", "about",
            "how", "what", "why", "when", "where", "who", "which", "i", "you",
            "me", "my", "your", "it", "its", "this", "that", "these", "those"
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Also extract capitalized phrases (likely important)
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        keywords.extend([p.lower() for p in cap_phrases])
        
        return keywords
    
    def _get_embedding(self, text: str, cache_key: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get embedding for text, using cache if available.
        
        Args:
            text: Text to embed
            cache_key: Optional cache key
            
        Returns:
            Embedding vector or None
        """
        if not self.embedding_model:
            return None
        
        # Check cache
        if cache_key and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            
            # Cache if key provided
            if cache_key and self.config.cache_embeddings:
                self.embedding_cache[cache_key] = embedding
                
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to 0-1
    
    def _load_user_preferences(self):
        """Load user preferences (starred/pinned items)."""
        pref_file = Path("models/mesh_contexts/user_preferences.json")
        if pref_file.exists():
            try:
                with open(pref_file, 'r') as f:
                    self.user_preferences = json.load(f)
                logger.info(f"Loaded user preferences for {len(self.user_preferences)} users")
            except Exception as e:
                logger.error(f"Failed to load user preferences: {e}")
    
    def _save_user_preferences(self):
        """Save user preferences."""
        pref_file = Path("models/mesh_contexts/user_preferences.json")
        try:
            with open(pref_file, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
    
    def _is_starred(self, user_id: Optional[str], item_key: str) -> bool:
        """Check if item is starred by user."""
        if not user_id or user_id not in self.user_preferences:
            return False
        return self.user_preferences[user_id].get(item_key, {}).get("starred", False)
    
    def _get_user_weight(self, user_id: Optional[str], item_key: str) -> float:
        """Get user weight for item."""
        if not user_id or user_id not in self.user_preferences:
            return 0.0
        return self.user_preferences[user_id].get(item_key, {}).get("weight", 0.0)
    
    def star_item(self, user_id: str, item_key: str, weight: float = 1.0):
        """
        Star/pin an item for a user (for future UI integration).
        
        Args:
            user_id: User identifier
            item_key: Item key (concept name or intent ID)
            weight: User weight (0-1)
        """
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id][item_key] = {
            "starred": True,
            "weight": min(1.0, max(0.0, weight)),
            "starred_at": datetime.now().isoformat()
        }
        
        self._save_user_preferences()
        logger.info(f"User {user_id} starred '{item_key}' with weight {weight}")
    
    def unstar_item(self, user_id: str, item_key: str):
        """
        Unstar/unpin an item for a user.
        
        Args:
            user_id: User identifier
            item_key: Item key
        """
        if user_id in self.user_preferences and item_key in self.user_preferences[user_id]:
            del self.user_preferences[user_id][item_key]
            self._save_user_preferences()
            logger.info(f"User {user_id} unstarred '{item_key}'")
    
    def _load_embedding_cache(self):
        """Load pre-computed embeddings from cache."""
        cache_file = self.cache_dir / "embeddings.npz"
        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                self.embedding_cache = dict(data.items())
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.error(f"Failed to load embedding cache: {e}")
    
    def save_embedding_cache(self):
        """Save embeddings to cache."""
        if not self.config.cache_embeddings or not self.embedding_cache:
            return
        
        cache_file = self.cache_dir / "embeddings.npz"
        try:
            np.savez_compressed(cache_file, **self.embedding_cache)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def precompute_embeddings(self, mesh_context: Dict[str, Any]):
        """
        Precompute embeddings for all context items.
        
        Args:
            mesh_context: Mesh context to precompute
        """
        if not self.embedding_model:
            return
        
        count = 0
        
        # Precompute for concepts
        for concept_list in [
            mesh_context.get("personal_concepts", []),
            *mesh_context.get("team_concepts", {}).values()
        ]:
            for concept in concept_list:
                text = f"{concept.get('name', '')} {concept.get('summary', '')}"
                key = concept.get("name")
                if key and key not in self.embedding_cache:
                    self._get_embedding(text, cache_key=key)
                    count += 1
        
        # Precompute for intents
        for intent in mesh_context.get("open_intents", []):
            text = intent.get("description", "")
            key = intent.get("id")
            if key and key not in self.embedding_cache:
                self._get_embedding(text, cache_key=key)
                count += 1
        
        if count > 0:
            self.save_embedding_cache()
            logger.info(f"Precomputed {count} new embeddings")
    
    def _count_items(self, context: Dict[str, Any]) -> int:
        """Count total items in context."""
        count = 0
        count += len(context.get("personal_concepts", []))
        count += len(context.get("open_intents", []))
        for concepts in context.get("team_concepts", {}).values():
            count += len(concepts)
        if context.get("recent_activity"):
            count += 1
        return count
    
    def _log_filtering_stats(self,
                            original: Dict[str, Any],
                            filtered: Dict[str, Any],
                            selected: List[ScoredContextItem]):
        """Log filtering statistics."""
        original_count = self._count_items(original)
        filtered_count = self._count_items(filtered)
        
        logger.info(f"Context filtering: {original_count} → {filtered_count} items")
        logger.info(f"  Mode: {self.config.mode.value}")
        logger.info(f"  Top scores: {[round(item.score, 3) for item in selected[:5]]}")
        
        if selected:
            logger.debug("Top items selected:")
            for item in selected[:3]:
                logger.debug(f"  - {item.context_type.value}: score={item.score:.3f}, reasons={item.reasons}")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_global_filter: Optional[ContextFilter] = None

def get_global_filter(config: Optional[FilterConfig] = None) -> ContextFilter:
    """
    Get or create global filter instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        Global ContextFilter instance
    """
    global _global_filter
    
    if _global_filter is None:
        _global_filter = ContextFilter(config)
    
    return _global_filter

def filter_context_for_prompt(mesh_context: Dict[str, Any],
                             prompt: str,
                             user_id: Optional[str] = None,
                             mode: Optional[WeightingMode] = None) -> Dict[str, Any]:
    """
    Quick function to filter context for a prompt.
    
    Args:
        mesh_context: Full mesh context
        prompt: User's prompt
        user_id: User identifier
        mode: Optional mode override
        
    Returns:
        Filtered context
    """
    config = FilterConfig(mode=mode) if mode else None
    filter = get_global_filter(config)
    return filter.filter_relevant_context(mesh_context, prompt, user_id)

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "WeightingMode",
    "ContextType",
    "ScoredContextItem",
    "FilterConfig",
    "ContextFilter",
    "get_global_filter",
    "filter_context_for_prompt"
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
    
    # Test context filtering
    test_context = {
        "user_id": "test_user",
        "timestamp": datetime.now().isoformat(),
        "personal_concepts": [
            {"name": "Alpha Protocol", "summary": "Security protocol for data encryption", "score": 0.9},
            {"name": "Beta Algorithm", "summary": "Optimization algorithm", "score": 0.7},
            {"name": "Project X", "summary": "Main development project", "score": 0.8},
            {"name": "Database Schema", "summary": "PostgreSQL database design", "score": 0.5},
            {"name": "API Gateway", "summary": "Microservices API management", "score": 0.4}
        ],
        "open_intents": [
            {"id": "opt_001", "description": "Optimize Alpha Protocol performance", "priority": "high"},
            {"id": "doc_002", "description": "Complete Project X documentation", "priority": "normal"},
            {"id": "bug_003", "description": "Fix database connection pooling", "priority": "low"}
        ],
        "recent_activity": "Working on Alpha Protocol optimization and API Gateway configuration",
        "team_concepts": {
            "ProjectX": [
                {"name": "Sprint Planning", "summary": "Q4 sprint goals", "score": 0.6}
            ]
        }
    }
    
    # Test different prompts
    test_prompts = [
        "Tell me about Alpha Protocol",
        "How can I optimize performance?",
        "What's the status of Project X?",
        "Help with database issues",
        "Something completely unrelated to work"
    ]
    
    # Create filter with hybrid mode
    filter = ContextFilter(FilterConfig(mode=WeightingMode.HYBRID))
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        
        filtered = filter.filter_relevant_context(test_context, prompt, "test_user")
        
        print(f"Original items: {filter._count_items(test_context)}")
        print(f"Filtered items: {filter._count_items(filtered)}")
        
        if "filter_stats" in filtered:
            print(f"Top scores: {filtered['filter_stats']['top_scores']}")
        
        # Show what was included
        if "personal_concepts" in filtered:
            print(f"Concepts: {[c['name'] for c in filtered['personal_concepts']]}")
        if "open_intents" in filtered:
            print(f"Intents: {[i['id'] for i in filtered['open_intents']]}")
