"""
Kaizen-specific fixes to apply

This file contains the exact code to add to kaizen.py to fix the remaining issues.
"""

# 1. Add this import guard for numpy (add after other imports):
"""
# Optional dependencies with guards
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.debug("NumPy not available, some features will be disabled")
"""

# 2. Fix the analyze_query_patterns method to guard numpy usage:
"""
async def _analyze_query_patterns(self, queries: List[Dict[str, Any]]) -> List[LearningInsight]:
    '''Analyze patterns in queries - with numpy guard'''
    insights = []
    
    # Pattern frequency analysis
    pattern_counts = defaultdict(int)
    for query in queries:
        # Extract patterns (simple word frequency for now)
        words = query.get("content", "").lower().split()
        for word in words:
            if len(word) > 3:  # Skip short words
                pattern_counts[word] += 1
    
    # Find significant patterns
    if pattern_counts:
        if NUMPY_AVAILABLE:
            # Use numpy for statistical analysis
            counts = np.array(list(pattern_counts.values()))
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            threshold = mean_count + std_count
        else:
            # Fallback to pure Python
            counts = list(pattern_counts.values())
            mean_count = sum(counts) / len(counts) if counts else 0
            # Simple std deviation calculation
            variance = sum((x - mean_count) ** 2 for x in counts) / len(counts) if counts else 0
            std_count = variance ** 0.5
            threshold = mean_count + std_count
        
        # Create insights for significant patterns
        for pattern, count in pattern_counts.items():
            if count > threshold:
                insights.append(LearningInsight(
                    insight_type="query_pattern",
                    description=f"Frequent query pattern detected: '{pattern}' ({count} occurrences)",
                    confidence=min(count / len(queries), 1.0),
                    data={"pattern": pattern, "count": count},
                    timestamp=datetime.utcnow()
                ))
    
    return insights
"""

# 3. Add the kaizen_health critic at the END of kaizen.py:
"""
# --- Critic registration ---
try:
    from kha.meta_genome.critics.critic_hub import critic, evaluate
    
    @critic("kaizen_health")
    def kaizen_health(report):
        '''Score Kaizen's own success rate (insights applied / generated).'''
        ratio = report.get("kaizen_success_rate", 1.0)
        return ratio, ratio >= 0.70
    
    # Make evaluate available for use in the class
    CRITIC_HUB_AVAILABLE = True
    
except ImportError:
    logger.debug("Critic hub not available, health monitoring disabled")
    CRITIC_HUB_AVAILABLE = False
    evaluate = lambda x: None  # No-op function
"""

# 4. Update run_analysis_cycle to call evaluate (add at the end of the method):
"""
# At the end of run_analysis_cycle, after applying insights:

# Calculate success rate for critic evaluation
if self.insights:
    recent_insights = self.insights[-50:]  # Last 50 insights
    applied_count = sum(1 for i in recent_insights if i.applied)
    success_rate = applied_count / len(recent_insights) if recent_insights else 1.0
else:
    success_rate = 1.0

# Submit to critic hub
critic_report = {
    "kaizen_success_rate": success_rate,
    "kaizen_total_insights": len(self.insights),
    "kaizen_applied_insights": sum(1 for i in self.insights if i.applied),
    "kaizen_performance_score": performance_score,
    "timestamp": datetime.utcnow().isoformat()
}

if CRITIC_HUB_AVAILABLE:
    try:
        evaluate(critic_report)
        logger.debug(f"Submitted critic report with success rate: {success_rate:.2f}")
    except Exception as e:
        logger.error(f"Failed to submit to critic hub: {e}")

# Log critic report to psi_archive for health tracking
if psi_archive:
    psi_archive.log_event("kaizen_health", critic_report)

# Also log analysis completion to psi_archive
psi_archive.log_event("kaizen_analysis_completed", {
    "insights_generated": len(new_insights),
    "total_insights": len(self.insights),
    "success_rate": success_rate,
    "performance_score": performance_score
})
"""

# 5. Update the default config to use environment variable:
"""
_default_config = {
    "analysis_interval": int(os.getenv("KAIZEN_ANALYSIS_INTERVAL", "300")),  # 5 min default
    "min_data_points": 10,
    "enable_auto_apply": False,
    "max_insights_stored": 10000,
    "confidence_threshold": 0.7,
    "max_insights_per_cycle": 5,
    "enable_clustering": False,
    "knowledge_base_path": None,
}
"""

# 6. Create a .env.example file for the create directory:
ENV_EXAMPLE = """# MCP Server Creator Environment Configuration

# Global default for all servers (5 minutes for development)
DEFAULT_ANALYSIS_INTERVAL=300

# Kaizen-specific settings
KAIZEN_ANALYSIS_INTERVAL=300
KAIZEN_ENABLE_AUTO_APPLY=false
KAIZEN_CONFIDENCE_THRESHOLD=0.7

# Example server settings
EMPATHY_ANALYSIS_INTERVAL=600
EMPATHY_ENABLE_WATCHDOG=true
EMPATHY_ENABLE_CRITICS=true

# Intent server settings  
INTENT_ANALYSIS_INTERVAL=300
INTENT_WATCHDOG_TIMEOUT=60

# Production settings (uncomment for production)
# DEFAULT_ANALYSIS_INTERVAL=3600
# KAIZEN_ANALYSIS_INTERVAL=3600
# KAIZEN_ENABLE_AUTO_APPLY=true
"""
