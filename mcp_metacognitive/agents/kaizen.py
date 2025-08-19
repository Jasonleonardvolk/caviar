"""
Kaizen Agent - Continuous Improvement Module
===========================================

This module implements TORI's continuous learning and improvement system:
- Analyzes conversation logs and performance metrics
- Identifies patterns and areas for improvement
- Updates knowledge base and system parameters
- Runs autonomously in the background (always-on)
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import os
from pathlib import Path

from ..core.agent_registry import Agent
from ..core.psi_archive import psi_archive
from ..core.state_manager import state_manager

# Import new components
try:
    from .kaizen_config import KaizenConfig, load_config
    from .kaizen_metrics import metrics_exporter
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    
try:
    import backoff
    BACKOFF_AVAILABLE = True
except ImportError:
    BACKOFF_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Tracks performance metrics for analysis"""
    response_times: deque = field(default_factory=lambda: deque(maxlen=10000))
    error_rates: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    query_patterns: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    success_rates: Dict[str, float] = field(default_factory=dict)
    consciousness_levels: deque = field(default_factory=lambda: deque(maxlen=10000))
    user_satisfaction_scores: deque = field(default_factory=lambda: deque(maxlen=10000))
    total_queries: int = 0  # Track total queries explicitly
    
    def add_response_time(self, time: float):
        self.response_times.append(time)
    
    def add_error(self, error_type: str):
        self.error_rates[error_type] += 1
    
    def add_query_pattern(self, pattern: str):
        self.query_patterns[pattern] += 1
    
    def add_consciousness_level(self, level: float):
        self.consciousness_levels.append(level)
    
    def add_satisfaction_score(self, score: float):
        self.user_satisfaction_scores.append(score)
    
    def get_average_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    def get_error_rate(self) -> float:
        total_errors = sum(self.error_rates.values())
        return total_errors / self.total_queries if self.total_queries > 0 else 0.0

@dataclass
class LearningInsight:
    """Represents a learning insight discovered by Kaizen"""
    insight_type: str
    description: str
    confidence: float
    data: Dict[str, Any]
    timestamp: datetime
    applied: bool = False

class KaizenImprovementEngine(Agent):
    """
    Kaizen - TORI's continuous improvement and learning module
    Runs autonomously to analyze performance and improve over time
    """
    
    # Metadata for dynamic discovery
    _metadata = {
        "name": "kaizen",
        "description": "Continuous improvement engine that analyzes performance and generates insights",
        "enabled": True,
        "auto_start": True,
        "endpoints": [
            {"path": "/api/insights", "method": "GET", "description": "Get recent insights"},
            {"path": "/api/analyze", "method": "POST", "description": "Trigger analysis"}
        ],
        "dependencies": ["daniel"],  # Depends on Daniel for analyzing conversations
        "version": "1.0.0"
    }
    
    _default_config = {
        "analysis_interval": 3600,
        "min_data_points": 10,
        "enable_auto_apply": False,
        "max_insights_stored": 10000,  # Cap insights in memory
        "confidence_threshold": 0.7,  # Minimum confidence for insights
        "max_insights_per_cycle": 5,  # Max insights to generate per cycle
        "enable_clustering": False,  # Disabled by default - can be enabled if sklearn installed
        "knowledge_base_path": None,  # Will be set to absolute path in __init__
        "metrics_retention_days": 7,  # How long to keep metrics
        "performance_threshold": {
            "response_time": 2.0,  # seconds
            "error_rate": 0.05,  # 5%
            "consciousness_level": 0.4  # minimum healthy consciousness
        }
    }
    
    def __init__(self, name: str = "kaizen", config: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        
        # Load configuration with Pydantic if available
        if CONFIG_AVAILABLE and config is None:
            self.pydantic_config = KaizenConfig.from_env()
            self.config = self.pydantic_config.to_legacy_dict()
        elif CONFIG_AVAILABLE and isinstance(config, dict):
            # Validate dict config with Pydantic
            self.pydantic_config = KaizenConfig(**config)
            self.config = self.pydantic_config.to_legacy_dict()
        else:
            # Fallback to legacy config handling
            base_config = self._get_config_with_env()
            if config:
                base_config.update(config)
            self.config = base_config
            self.pydantic_config = None
        
        self.metrics = PerformanceMetrics()
        self.insights: List[LearningInsight] = []
        self.knowledge_base: Dict[str, Any] = {}
        self.improvement_history: List[Dict[str, Any]] = []
        self.is_running = False
        self.analysis_task = None
        
        # Thread safety for insights
        self._insight_lock = asyncio.Lock()
        
        # Initialize knowledge base path with absolute path
        if self.config.get("knowledge_base_path") is None:
            default_kb_path = Path(__file__).parent.parent / "data" / "kaizen_kb.json"
            self.config["knowledge_base_path"] = str(default_kb_path)
        
        self.kb_path = Path(self.config["knowledge_base_path"])
        self._load_knowledge_base()
        
        # Initialize metrics exporter if available
        if CONFIG_AVAILABLE and self.config.get("enable_prometheus"):
            metrics_exporter.set_kaizen_engine(self)
            logger.info("Prometheus metrics export enabled")
        
        # Initialize circuit breaker state
        self._gap_fill_failures = 0
        self._gap_fill_backoff_until = None
        
        # Log initialization
        psi_archive.log_event("kaizen_initialized", {
            "config": self.config,
            "knowledge_base_size": len(self.knowledge_base)
        })
    
    def _get_config_with_env(self) -> Dict[str, Any]:
        """Get configuration with environment overrides"""
        # Start with a copy of defaults to avoid mutation
        config = self._default_config.copy()
        
        # Override with environment variables
        if os.getenv("KAIZEN_ANALYSIS_INTERVAL"):
            config["analysis_interval"] = int(os.getenv("KAIZEN_ANALYSIS_INTERVAL"))
        if os.getenv("KAIZEN_MIN_DATA_POINTS"):
            config["min_data_points"] = int(os.getenv("KAIZEN_MIN_DATA_POINTS"))
        if os.getenv("KAIZEN_ENABLE_AUTO_APPLY"):
            config["enable_auto_apply"] = os.getenv("KAIZEN_ENABLE_AUTO_APPLY", "false").lower() == "true"
        if os.getenv("KAIZEN_ENABLE_CLUSTERING"):
            config["enable_clustering"] = os.getenv("KAIZEN_ENABLE_CLUSTERING", "false").lower() == "true"
        
        return config
    
    async def execute(self, command: str = "analyze", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Kaizen operations
        Commands: start, stop, analyze, get_insights, apply_insight
        """
        if command == "start":
            return await self.start_continuous_improvement()
        elif command == "stop":
            return await self.stop_continuous_improvement()
        elif command == "analyze":
            return await self.run_analysis_cycle()
        elif command == "get_insights":
            return await self.get_recent_insights(params.get("limit", 10) if params else 10)
        elif command == "apply_insight":
            return await self.apply_insight(params.get("insight_id") if params else None)
        else:
            return {"status": "error", "message": f"Unknown command: {command}"}
    
    async def start_continuous_improvement(self) -> Dict[str, Any]:
        """Start the continuous improvement loop"""
        if self.is_running:
            return {"status": "already_running", "message": "Kaizen is already running"}
        
        self.is_running = True
        self.analysis_task = asyncio.create_task(self._continuous_improvement_loop())
        
        psi_archive.log_event("kaizen_started", {
            "analysis_interval": self.config["analysis_interval"]
        })
        
        return {
            "status": "started",
            "message": "Kaizen continuous improvement started",
            "interval": self.config["analysis_interval"]
        }
    
    async def stop_continuous_improvement(self) -> Dict[str, Any]:
        """Stop the continuous improvement loop"""
        if not self.is_running:
            return {"status": "not_running", "message": "Kaizen is not running"}
        
        self.is_running = False
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        
        # Save knowledge base before stopping
        self._save_knowledge_base()
        
        psi_archive.log_event("kaizen_stopped", {
            "total_insights": len(self.insights),
            "knowledge_base_size": len(self.knowledge_base)
        })
        
        return {
            "status": "stopped",
            "message": "Kaizen continuous improvement stopped"
        }
    
    async def _continuous_improvement_loop(self):
        """Main loop for continuous improvement"""
        logger.info("Kaizen continuous improvement loop started")
        
        while self.is_running:
            try:
                # Wait for the configured interval
                await asyncio.sleep(self.config["analysis_interval"])
                
                # Run analysis cycle
                await self.run_analysis_cycle()
                
                # Auto-apply high confidence insights if enabled
                if self.config.get("enable_auto_apply"):
                    await self._auto_apply_insights()
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Kaizen loop: {e}")
                psi_archive.log_event("kaizen_error", {"error": str(e)})
    
    async def run_analysis_cycle(self) -> Dict[str, Any]:
        """Run a complete analysis cycle"""
        start_time = datetime.utcnow()
        
        # Collect recent events from PsiArchive
        recent_events = await self._collect_recent_events()
        
        if len(recent_events) < self.config["min_data_points"]:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {self.config['min_data_points']} data points"
            }
        
        # Update metrics from events
        self._update_metrics_from_events(recent_events)
        
        # Perform various analyses
        insights = []
        
        # 1. Performance analysis
        perf_insights = await self._analyze_performance()
        insights.extend(perf_insights)
        
        # 2. Error pattern analysis
        error_insights = await self._analyze_error_patterns()
        insights.extend(error_insights)
        
        # 3. Query pattern analysis
        pattern_insights = await self._analyze_query_patterns()
        insights.extend(pattern_insights)
        
        # 4. Consciousness stability analysis
        consciousness_insights = await self._analyze_consciousness_stability()
        insights.extend(consciousness_insights)
        
        # 5. Response quality analysis (inferred from patterns)
        quality_insights = await self._analyze_response_quality(recent_events)
        insights.extend(quality_insights)
        
        # Sort by confidence and limit
        insights.sort(key=lambda x: x.confidence, reverse=True)
        insights = insights[:self.config["max_insights_per_cycle"]]
        
        # Store insights with thread safety
        async with self._insight_lock:
            self.insights.extend(insights)
            
            # Cap total insights to prevent unbounded growth
            max_insights = self.config.get("max_insights_stored", 10000)
            if len(self.insights) > max_insights:
                # Keep the most recent insights
                self.insights = sorted(
                    self.insights,
                    key=lambda x: x.timestamp,
                    reverse=True
                )[:max_insights]
        
        # Update knowledge base
        self._update_knowledge_base_from_insights(insights)
        
        # Save knowledge base
        self._save_knowledge_base()
        
        # Log completion
        analysis_time = (datetime.utcnow() - start_time).total_seconds()
        psi_archive.log_event("kaizen_analysis_completed", {
            "events_analyzed": len(recent_events),
            "insights_generated": len(insights),
            "analysis_time": analysis_time,
            "top_insight": insights[0].description if insights else None
        })
        
        return {
            "status": "completed",
            "events_analyzed": len(recent_events),
            "insights_generated": len(insights),
            "analysis_time": analysis_time,
            "insights": [self._serialize_insight(i) for i in insights]
        }
    
    async def _collect_recent_events(self) -> List[Dict[str, Any]]:
        """Collect recent events from PsiArchive"""
        # Get events from the last analysis interval
        # Run in executor to avoid blocking if psi_archive does I/O
        loop = asyncio.get_event_loop()
        events = await loop.run_in_executor(
            None,
            lambda: psi_archive.get_recent_events(limit=1000)
        )
        
        # Filter relevant events
        relevant_event_types = [
            "daniel_query_received",
            "daniel_query_completed",
            "daniel_error",
            "consciousness_update",
            "tool_invoked",
            "filter_applied",
            "critics_result"  # Added to analyze critic decisions
        ]
        
        filtered_events = [
            e for e in events 
            if e.get("event") in relevant_event_types
        ]
        
        return filtered_events
    
    def _update_metrics_from_events(self, events: List[Dict[str, Any]]):
        """Update performance metrics from events"""
        for event in events:
            event_type = event.get("event")
            data = event.get("data", {})
            
            if event_type == "daniel_query_completed":
                # Track response time
                if "processing_time" in data:
                    self.metrics.add_response_time(data["processing_time"])
                
                # Track consciousness level
                if "consciousness_level" in data:
                    self.metrics.add_consciousness_level(data["consciousness_level"])
            
            elif event_type == "daniel_error":
                # Track errors
                error_type = data.get("error", "unknown")
                self.metrics.add_error(error_type)
            
            elif event_type == "daniel_query_received":
                # Track total queries
                self.metrics.total_queries += 1
                
                # Analyze query patterns
                query = data.get("query", "")
                pattern = self._extract_query_pattern(query)
                if pattern:
                    self.metrics.add_query_pattern(pattern)
            
            elif event_type == "critics_result":
                # Track critic decisions for gap analysis
                if data.get("accepted") is False:
                    # Analyze why critics rejected
                    scores = data.get("scores", {})
                    for critic, score in scores.items():
                        if score < 0.5:
                            self.metrics.add_error(f"critic_rejection_{critic}")
    
    def _extract_query_pattern(self, query: str) -> Optional[str]:
        """Extract pattern from query for analysis"""
        # Simple pattern extraction - can be enhanced
        if "?" in query:
            if any(word in query.lower() for word in ["what", "why", "how", "when", "where", "who"]):
                return "question_wh"
            else:
                return "question_other"
        elif any(word in query.lower() for word in ["create", "generate", "write", "make"]):
            return "creative_request"
        elif any(word in query.lower() for word in ["analyze", "evaluate", "assess", "review"]):
            return "analysis_request"
        else:
            return "statement_or_other"
    
    async def _analyze_performance(self) -> List[LearningInsight]:
        """Analyze system performance metrics"""
        insights = []
        
        # Check response time
        avg_response_time = self.metrics.get_average_response_time()
        threshold = self.config["performance_threshold"]["response_time"]
        
        if avg_response_time > threshold:
            insights.append(LearningInsight(
                insight_type="performance",
                description=f"Average response time ({avg_response_time:.2f}s) exceeds threshold ({threshold}s)",
                confidence=0.9,
                data={
                    "metric": "response_time",
                    "current": avg_response_time,
                    "threshold": threshold,
                    "recommendation": "Consider optimizing query processing or caching"
                },
                timestamp=datetime.utcnow()
            ))
        
        # Check error rate
        error_rate = self.metrics.get_error_rate()
        error_threshold = self.config["performance_threshold"]["error_rate"]
        
        if error_rate > error_threshold:
            insights.append(LearningInsight(
                insight_type="reliability",
                description=f"Error rate ({error_rate:.2%}) exceeds threshold ({error_threshold:.2%})",
                confidence=0.95,
                data={
                    "metric": "error_rate",
                    "current": error_rate,
                    "threshold": error_threshold,
                    "top_errors": dict(sorted(
                        self.metrics.error_rates.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3])
                },
                timestamp=datetime.utcnow()
            ))
        
        return insights
    
    async def _analyze_error_patterns(self) -> List[LearningInsight]:
        """Analyze error patterns for systemic issues"""
        insights = []
        
        if not self.metrics.error_rates:
            return insights
        
        # Find most common errors
        top_errors = sorted(
            self.metrics.error_rates.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for error_type, count in top_errors:
            if count > 5:  # Threshold for concern
                insight = LearningInsight(
                    insight_type="error_pattern",
                    description=f"Frequent error pattern detected: {error_type} ({count} occurrences)",
                    confidence=0.8,
                    data={
                        "error_type": error_type,
                        "count": count,
                        "recommendation": f"Investigate root cause of {error_type} errors"
                    },
                    timestamp=datetime.utcnow()
                )
                insights.append(insight)
                
                # Check for gap-fill opportunities
                if insight.insight_type == "gap" or "gap" in error_type.lower():
                    # Trigger paper search for gap-fill
                    try:
                        from ..core.mcp_bridge import mcp_bridge
                        mcp_bridge.dispatch("paper_search", {"query": insight.description})
                        logger.info(f"Triggered paper search for gap: {insight.description}")
                    except Exception as e:
                        logger.error(f"Failed to trigger paper search: {e}")
        
        return insights
    
    async def _analyze_query_patterns(self) -> List[LearningInsight]:
        """Analyze query-pattern distribution (with optional clustering if enabled)"""
        insights = []
        
        if len(self.metrics.query_patterns) < 10:
            return insights
        
        # Get pattern distribution
        total_queries = sum(self.metrics.query_patterns.values())
        pattern_distribution = {
            pattern: count / total_queries 
            for pattern, count in self.metrics.query_patterns.items()
        }
        
        # Check if real clustering is requested and available
        if self.config.get("enable_clustering"):
            try:
                # Lazy import for optional dependency
                from sklearn.cluster import KMeans
                from sklearn.feature_extraction.text import TfidfVectorizer
                import numpy as np
                
                # Get query samples for each pattern
                # Note: This is a simplified example - in production you'd store actual queries
                patterns = list(self.metrics.query_patterns.keys())
                if len(patterns) >= 3:  # Need at least 3 patterns for meaningful clustering
                    # Create simple feature vectors based on pattern names
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform(patterns)
                    
                    # Perform K-means clustering
                    n_clusters = min(3, len(patterns) // 2)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X)
                    
                    # Analyze clusters
                    for i in range(n_clusters):
                        cluster_patterns = [p for j, p in enumerate(patterns) if clusters[j] == i]
                        cluster_size = sum(self.metrics.query_patterns[p] for p in cluster_patterns)
                        cluster_ratio = cluster_size / total_queries
                        
                        if cluster_ratio > 0.2:  # Significant cluster
                            insights.append(LearningInsight(
                                insight_type="usage_cluster",
                                description=f"Query cluster detected: {', '.join(cluster_patterns[:3])} ({cluster_ratio:.1%} of queries)",
                                confidence=0.8,
                                data={
                                    "cluster_patterns": cluster_patterns,
                                    "ratio": cluster_ratio,
                                    "recommendation": f"Optimize for this query cluster"
                                },
                                timestamp=datetime.utcnow()
                            ))
                            
            except ImportError:
                logger.warning("Clustering requested but sklearn not available - falling back to ratio analysis. Install scikit-learn to enable clustering.")
                # Ensure we still return insights even without clustering
            except Exception as e:
                logger.error(f"Clustering failed: {e}")
        
        # Always do ratio-based analysis as fallback or primary method
        dominant_patterns = [
            (pattern, ratio) 
            for pattern, ratio in pattern_distribution.items() 
            if ratio > 0.3
        ]
        
        for pattern, ratio in dominant_patterns:
            insights.append(LearningInsight(
                insight_type="usage_pattern",
                description=f"Dominant query pattern: {pattern} ({ratio:.1%} of queries)",
                confidence=0.75,
                data={
                    "pattern": pattern,
                    "ratio": ratio,
                    "recommendation": f"Optimize responses for {pattern} queries"
                },
                timestamp=datetime.utcnow()
            ))
        
        return insights
    
    async def _analyze_consciousness_stability(self) -> List[LearningInsight]:
        """Analyze consciousness level stability"""
        insights = []
        
        if len(self.metrics.consciousness_levels) < 10:
            return insights
        
        # Calculate consciousness statistics
        avg_consciousness = statistics.mean(self.metrics.consciousness_levels)
        std_consciousness = statistics.stdev(self.metrics.consciousness_levels)
        min_consciousness = min(self.metrics.consciousness_levels)
        
        threshold = self.config["performance_threshold"]["consciousness_level"]
        
        if avg_consciousness < threshold:
            insights.append(LearningInsight(
                insight_type="consciousness",
                description=f"Low average consciousness level ({avg_consciousness:.2f})",
                confidence=0.85,
                data={
                    "average": avg_consciousness,
                    "std_dev": std_consciousness,
                    "minimum": min_consciousness,
                    "threshold": threshold,
                    "recommendation": "Increase cognitive stimulation or adjust parameters"
                },
                timestamp=datetime.utcnow()
            ))
        
        if std_consciousness > 0.2:  # High variability
            insights.append(LearningInsight(
                insight_type="stability",
                description=f"High consciousness variability (σ={std_consciousness:.2f})",
                confidence=0.7,
                data={
                    "std_dev": std_consciousness,
                    "recommendation": "Stabilize cognitive processing parameters"
                },
                timestamp=datetime.utcnow()
            ))
        
        return insights
    
    async def _analyze_response_quality(self, events: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Analyze response quality from indirect signals"""
        insights = []
        
        # Look for patterns indicating user dissatisfaction
        # (since we don't have direct feedback)
        clarification_requests = 0
        repeated_queries = 0
        short_sessions = 0
        
        # Simple heuristic analysis
        query_history = []
        for event in events:
            if event.get("event") == "daniel_query_received":
                query = event.get("data", {}).get("query", "").lower()
                
                # Check for clarification patterns
                if any(word in query for word in ["what do you mean", "clarify", "explain", "confused"]):
                    clarification_requests += 1
                
                # Check for repeated similar queries
                if query_history and any(
                    self._query_similarity(query, prev) > 0.8 
                    for prev in query_history[-3:]
                ):
                    repeated_queries += 1
                
                query_history.append(query)
        
        # Generate insights based on patterns
        if clarification_requests > 5:
            insights.append(LearningInsight(
                insight_type="clarity",
                description=f"High clarification requests ({clarification_requests})",
                confidence=0.75,
                data={
                    "clarification_requests": clarification_requests,
                    "recommendation": "Improve response clarity and completeness"
                },
                timestamp=datetime.utcnow()
            ))
        
        if repeated_queries > 3:
            insights.append(LearningInsight(
                insight_type="effectiveness",
                description=f"Users repeating similar queries ({repeated_queries} times)",
                confidence=0.7,
                data={
                    "repeated_queries": repeated_queries,
                    "recommendation": "Improve initial response quality"
                },
                timestamp=datetime.utcnow()
            ))
        
        return insights
    
    def _query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        # Simple word overlap similarity
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Guard against division by zero
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    async def _auto_apply_insights(self):
        """Automatically apply high-confidence insights with timeout"""
        async with self._insight_lock:
            applicable_insights = [
                i for i in self.insights 
                if not i.applied and i.confidence >= 0.9
            ]
        
        for insight in applicable_insights:
            try:
                # Apply with timeout to prevent blocking
                result = await asyncio.wait_for(
                    self.apply_insight(insight),
                    timeout=30.0  # 30 second timeout
                )
                if result.get("status") == "success":
                    async with self._insight_lock:
                        insight.applied = True
                    logger.info(f"Auto-applied insight: {insight.description}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout auto-applying insight: {insight.description}")
            except Exception as e:
                logger.error(f"Failed to auto-apply insight: {e}")
    
    async def apply_insight(self, insight: Union[LearningInsight, str]) -> Dict[str, Any]:
        """Apply a specific insight to improve the system"""
        async with self._insight_lock:
            if isinstance(insight, str):
                # Find insight by ID or description
                insight = next(
                    (i for i in self.insights if str(id(i)) == insight or i.description == insight),
                    None
                )
        
        if not insight:
            return {"status": "error", "message": "Insight not found"}
        
        # Apply based on insight type
        if insight.insight_type == "performance":
            # Adjust performance parameters
            self.knowledge_base["performance_adjustments"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "adjustments": insight.data.get("recommendation")
            }
            
        elif insight.insight_type == "usage_pattern":
            # Store pattern optimization
            pattern = insight.data.get("pattern")
            self.knowledge_base[f"pattern_optimization_{pattern}"] = {
                "pattern": pattern,
                "ratio": insight.data.get("ratio"),
                "optimization": "prioritize_processing"
            }
        
        # Mark as applied
        async with self._insight_lock:
            insight.applied = True
        
        # Log application
        psi_archive.log_event("kaizen_insight_applied", {
            "insight_type": insight.insight_type,
            "description": insight.description,
            "confidence": insight.confidence
        })
        
        return {
            "status": "success",
            "message": f"Applied insight: {insight.description}"
        }
    
    async def get_recent_insights(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent insights"""
        async with self._insight_lock:
            recent_insights = sorted(
                self.insights,
                key=lambda x: x.timestamp,
                reverse=True
            )[:limit]
        
        return {
            "status": "success",
            "insights": [self._serialize_insight(i) for i in recent_insights],
            "total_insights": len(self.insights),
            "applied_insights": sum(1 for i in self.insights if i.applied)
        }
    
    def _serialize_insight(self, insight: LearningInsight) -> Dict[str, Any]:
        """Serialize insight for external use"""
        return {
            "id": str(id(insight)),
            "type": insight.insight_type,
            "description": insight.description,
            "confidence": insight.confidence,
            "data": insight.data,
            "timestamp": insight.timestamp.isoformat(),
            "applied": insight.applied
        }
    
    def _update_knowledge_base_from_insights(self, insights: List[LearningInsight]):
        """Update knowledge base with new insights"""
        for insight in insights:
            key = f"{insight.insight_type}_{insight.timestamp.timestamp()}"
            self.knowledge_base[key] = {
                "description": insight.description,
                "confidence": insight.confidence,
                "data": insight.data,
                "timestamp": insight.timestamp.isoformat()
            }
    
    def _cleanup_old_metrics(self):
        """Clean up metrics older than retention period"""
        retention_days = self.config.get("metrics_retention_days", 7)
        cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
        
        # Clean up insights by time and cap
        async def cleanup_insights():
            async with self._insight_lock:
                self.insights = [
                    i for i in self.insights 
                    if i.timestamp > cutoff_time
                ]
                
                # Also apply the max cap
                max_insights = self.config.get("max_insights_stored", 10000)
                if len(self.insights) > max_insights:
                    self.insights = sorted(
                        self.insights,
                        key=lambda x: x.timestamp,
                        reverse=True
                    )[:max_insights]
        
        # Run cleanup in event loop
        asyncio.create_task(cleanup_insights())
        
        # Clean up old error counts periodically
        if self.metrics.total_queries > 100000:
            # Reset counters when we have too much historical data
            logger.info("Resetting performance metrics after 100k queries")
            self.metrics.error_rates.clear()
            self.metrics.query_patterns.clear()
            self.metrics.total_queries = 0
            
        # Limit success_rates dict size
        if len(self.metrics.success_rates) > 1000:
            # Keep only the most recent 1000 entries
            recent_keys = sorted(self.metrics.success_rates.keys())[-1000:]
            self.metrics.success_rates = {
                k: self.metrics.success_rates[k] 
                for k in recent_keys
            }
    
    def _load_knowledge_base(self):
        """Load knowledge base from disk"""
        try:
            if self.kb_path.exists():
                with self.kb_path.open('r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} entries")
            else:
                logger.info(f"Knowledge base not found at {self.kb_path}, starting fresh")
                self.knowledge_base = {}
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            self.knowledge_base = {}
    
    def _save_knowledge_base(self):
        """Save knowledge base to disk with atomic write"""
        try:
            kb_path = Path(self.kb_path)
            kb_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            tmp_path = kb_path.with_suffix('.tmp')
            with tmp_path.open('w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            
            # Atomic replace (works on POSIX & Windows)
            tmp_path.replace(kb_path)
            
            logger.debug(f"Saved knowledge base with {len(self.knowledge_base)} entries (atomic write)")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Kaizen improvement engine shutting down...")
        
        # Stop continuous improvement
        if self.is_running:
            await self.stop_continuous_improvement()
        
        # Save final state
        self._save_knowledge_base()
        
        # Log final statistics
        psi_archive.log_event("kaizen_shutdown", {
            "total_insights": len(self.insights),
            "applied_insights": sum(1 for i in self.insights if i.applied),
            "knowledge_base_size": len(self.knowledge_base),
            "avg_response_time": self.metrics.get_average_response_time(),
            "error_rate": self.metrics.get_error_rate()
        })

# Export
__all__ = ['KaizenImprovementEngine', 'PerformanceMetrics', 'LearningInsight']
