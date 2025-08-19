# Entropy pruning configuration - EQUAL ACCESS FOR ALL USERS
ENTROPY_CONFIG = {
    "max_diverse_concepts": None,  # No artificial limits - return all pure diverse concepts
    "entropy_threshold": 0.001,   # Very low threshold to maximize concept retention
    "similarity_threshold": 0.85, # Keep semantic diversity - remove >85% similar concepts
    "enable_categories": True,     # Use category-aware pruning for balanced coverage
    "concepts_per_category": None  # No category limits - return all diverse concepts per category
}