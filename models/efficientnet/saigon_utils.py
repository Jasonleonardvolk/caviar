"""
Saigon Utilities: Mesh relationship templates and helper functions
================================================================

Provides mesh-to-text conversion templates and utility functions for
the Saigon character-level language generation system.
"""

import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger("saigon_utils")

# Core mesh relationship templates
TEMPLATES = {
    "implies": "{A} implies {B}, indicating that the logical structure connects these concepts through necessity.",
    "supports": "{B} supports {A} by providing foundational evidence and reinforcing the conceptual framework.",
    "contradicts": "{A} contradicts {B} because their fundamental principles operate in opposing directions.",
    "extends": "{B} extends {A}, offering additional layers of complexity and semantic depth.",
    "relates_to": "{A} relates to {B} through shared semantic patterns and conceptual adjacency.",
    "derives_from": "{B} derives from {A}, inheriting core properties while developing new characteristics.",
    "enables": "{A} enables {B} by creating the necessary conditions for conceptual emergence.",
    "refines": "{B} refines {A} through precision and enhanced semantic resolution."
}

# Default template for unknown relations
DEFAULT_TEMPLATE = "{A} connects to {B} through dynamic conceptual linkage."


def validate_mesh_path(mesh_path: List[Dict[str, Any]]) -> bool:
    """
    Validate that a mesh path contains proper concept relationship structure.
    
    Args:
        mesh_path: List of concept relationship dictionaries
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(mesh_path, list):
        logger.warning("Mesh path must be a list")
        return False
    
    for i, node in enumerate(mesh_path):
        if not isinstance(node, dict):
            logger.warning(f"Node {i} must be a dictionary")
            return False
        
        required_fields = ['concept']
        for field in required_fields:
            if field not in node:
                logger.warning(f"Node {i} missing required field: {field}")
                return False
    
    return True


def format_mesh_relation(concept_a: str, concept_b: str, relation: str) -> str:
    """
    Format a single mesh relationship using templates.
    
    Args:
        concept_a: First concept
        concept_b: Second concept  
        relation: Relationship type
        
    Returns:
        str: Formatted relationship text
    """
    template = TEMPLATES.get(relation.lower(), DEFAULT_TEMPLATE)
    
    try:
        return template.format(A=concept_a.strip(), B=concept_b.strip())
    except KeyError as e:
        logger.warning(f"Template formatting error: {e}")
        return f"{concept_a} {relation} {concept_b}."


def mesh_to_text_enhanced(mesh_path: List[Dict[str, Any]]) -> str:
    """
    Enhanced mesh-to-text conversion with relationship templates.
    
    Args:
        mesh_path: List of concept relationship dictionaries
        
    Returns:
        str: Formatted text representation
    """
    if not validate_mesh_path(mesh_path):
        logger.error("Invalid mesh path structure")
        return "Invalid concept mesh structure."
    
    if not mesh_path:
        return "Empty concept mesh."
    
    sentences = []
    
    for i, node in enumerate(mesh_path):
        concept = node.get('concept', 'Unknown')
        relation = node.get('relation', 'relates_to')
        context = node.get('context', '')
        
        if i == 0:
            # First concept - establish context
            if context:
                sentence = f"In the context of {context}, {concept} serves as a foundational element."
            else:
                sentence = f"{concept} represents a core conceptual anchor."
        else:
            # Subsequent concepts - use relationships
            prev_concept = mesh_path[i-1].get('concept', 'Previous')
            sentence = format_mesh_relation(prev_concept, concept, relation)
        
        sentences.append(sentence)
    
    return " ".join(sentences)


def log_mesh_traversal(mesh_path: List[Dict[str, Any]], output_text: str, processing_time: float) -> Dict[str, Any]:
    """
    Create audit log for mesh traversal and text generation.
    
    Args:
        mesh_path: Original mesh path
        output_text: Generated text
        processing_time: Time taken for generation
        
    Returns:
        dict: Audit log data
    """
    audit_data = {
        "timestamp": time.time(),
        "mesh_path_length": len(mesh_path),
        "concepts": [node.get('concept', 'Unknown') for node in mesh_path],
        "relations": [node.get('relation', 'unknown') for node in mesh_path],
        "output_length": len(output_text),
        "processing_time": processing_time,
        "templates_used": list(set(node.get('relation', 'relates_to') for node in mesh_path))
    }
    
    logger.info(f"Mesh traversal: {audit_data['mesh_path_length']} concepts -> {audit_data['output_length']} chars in {processing_time:.3f}s")
    
    return audit_data


def get_available_relations() -> List[str]:
    """
    Get list of available relationship types.
    
    Returns:
        list: Available relation types
    """
    return list(TEMPLATES.keys())


def add_custom_template(relation: str, template: str) -> bool:
    """
    Add a custom relationship template.
    
    Args:
        relation: Relationship type name
        template: Template string with {A} and {B} placeholders
        
    Returns:
        bool: True if added successfully
    """
    if '{A}' not in template or '{B}' not in template:
        logger.error("Template must contain {A} and {B} placeholders")
        return False
    
    TEMPLATES[relation.lower()] = template
    logger.info(f"Added custom template for relation: {relation}")
    return True


def mesh_statistics(mesh_path: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about a mesh path.
    
    Args:
        mesh_path: Mesh path to analyze
        
    Returns:
        dict: Statistics
    """
    if not mesh_path:
        return {"error": "Empty mesh path"}
    
    concepts = [node.get('concept', '') for node in mesh_path]
    relations = [node.get('relation', 'unknown') for node in mesh_path]
    contexts = [node.get('context', '') for node in mesh_path if node.get('context')]
    
    return {
        "total_concepts": len(concepts),
        "unique_concepts": len(set(concepts)),
        "relation_types": len(set(relations)),
        "most_common_relation": max(set(relations), key=relations.count) if relations else None,
        "contexts_present": len(contexts),
        "average_concept_length": sum(len(c) for c in concepts) / len(concepts) if concepts else 0
    }
