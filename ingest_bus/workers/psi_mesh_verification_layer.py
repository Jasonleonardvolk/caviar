"""
ψMesh Verification Layer for TORI Document Ingestion
Implements concept cross-validation system with confidence scores and source matching

This is the integrity verification backbone ensuring extracted concepts are faithful to source content
"""

import json
import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
import re
import hashlib

from production_file_handlers import ParsedPayload

logger = logging.getLogger("tori-ingest.psi_mesh_verification")

class ConceptVerificationResult:
    """Result of concept verification process"""
    
    def __init__(self):
        self.concept_id: str = ""
        self.concept_name: str = ""
        self.integrity_score: float = 0.0
        self.confidence_score: float = 0.0
        self.verification_checks: Dict[str, Any] = {}
        self.source_matches: List[Dict[str, Any]] = []
        self.flagged_issues: List[str] = []
        self.verification_status: str = "pending"  # pending, verified, flagged, failed
        self.verification_metadata: Dict[str, Any] = {}

class PsiMeshVerificationLayer:
    """
    Advanced ψMesh verification system for concept integrity validation
    
    Ensures that extracted concepts are:
    1. Grounded in source text
    2. Semantically coherent
    3. Non-hallucinated
    4. Properly attributed with confidence scores
    """
    
    def __init__(self):
        self.verification_threshold = 0.75  # Minimum score for concept approval
        self.confidence_threshold = 0.60    # Minimum confidence for concept retention
        self.source_match_threshold = 0.80  # Minimum source matching score
        
        # Verification check weights
        self.check_weights = {
            'exact_text_match': 0.25,
            'keyword_presence': 0.20,
            'semantic_coherence': 0.20,
            'context_validation': 0.15,
            'source_attribution': 0.10,
            'frequency_analysis': 0.10
        }
        
        logger.info("ψMesh Verification Layer initialized")
        logger.info(f"Verification threshold: {self.verification_threshold}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
    
    def _identify_failed_issues(self, checks: Dict[str, Any]) -> List[str]:
        """Identify specific issues for failed concepts"""
        issues = []
        
        if checks.get('exact_text_match', {}).get('score', 0) < 0.3:
            issues.append("Concept name absent from source text")
        
        if checks.get('keyword_presence', {}).get('score', 0) < 0.3:
            issues.append("Keywords not found in source")
        
        if checks.get('semantic_coherence', {}).get('score', 0) < 0.3:
            issues.append("No semantic coherence with document")
        
        if checks.get('source_attribution', {}).get('score', 0) < 0.2:
            issues.append("Cannot attribute to any source segment")
        
        return issues
    
    def _generate_verification_summary(self, concept_results: List[ConceptVerificationResult]) -> Dict[str, Any]:
        """Generate summary of verification results"""
        if not concept_results:
            return {'status': 'no_concepts', 'message': 'No concepts to summarize'}
        
        verified_count = sum(1 for r in concept_results if r.verification_status == 'verified')
        flagged_count = sum(1 for r in concept_results if r.verification_status == 'flagged')
        failed_count = sum(1 for r in concept_results if r.verification_status == 'failed')
        
        avg_integrity = np.mean([r.integrity_score for r in concept_results])
        avg_confidence = np.mean([r.confidence_score for r in concept_results])
        
        return {
            'status': 'completed',
            'total_concepts': len(concept_results),
            'verified_count': verified_count,
            'flagged_count': flagged_count,
            'failed_count': failed_count,
            'verification_rate': verified_count / len(concept_results),
            'average_integrity_score': avg_integrity,
            'average_confidence_score': avg_confidence,
            'quality_assessment': self._assess_extraction_quality(verified_count, flagged_count, failed_count, avg_integrity)
        }
    
    def _assess_extraction_quality(self, verified: int, flagged: int, failed: int, avg_integrity: float) -> str:
        """Assess overall quality of concept extraction"""
        total = verified + flagged + failed
        if total == 0:
            return 'no_concepts'
        
        verification_rate = verified / total
        
        if verification_rate >= 0.9 and avg_integrity >= 0.85:
            return 'excellent'
        elif verification_rate >= 0.8 and avg_integrity >= 0.75:
            return 'good'
        elif verification_rate >= 0.6 and avg_integrity >= 0.65:
            return 'acceptable'
        elif verification_rate >= 0.4:
            return 'poor'
        else:
            return 'failed'
    
    def _generate_verification_recommendations(self, concept_results: List[ConceptVerificationResult], 
                                             verification_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on verification results"""
        recommendations = []
        
        quality = verification_results.get('verification_summary', {}).get('quality_assessment', 'unknown')
        verified_count = len(verification_results.get('verified_concepts', []))
        flagged_count = len(verification_results.get('flagged_concepts', []))
        failed_count = len(verification_results.get('failed_concepts', []))
        
        if quality == 'excellent':
            recommendations.append("Excellent concept extraction quality - proceed with confidence")
        elif quality == 'good':
            recommendations.append("Good concept extraction quality - minor refinements may help")
        elif quality == 'acceptable':
            recommendations.append("Acceptable quality - consider improving concept extraction methods")
        elif quality == 'poor':
            recommendations.append("Poor extraction quality - significant improvements needed")
        else:
            recommendations.append("Concept extraction failed - review extraction methodology")
        
        # Specific recommendations based on common issues
        common_issues = {}
        for result in concept_results:
            for issue in result.flagged_issues:
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        # Top issues
        sorted_issues = sorted(common_issues.items(), key=lambda x: x[1], reverse=True)
        
        for issue, count in sorted_issues[:3]:
            if count >= len(concept_results) * 0.3:  # Issue affects 30%+ of concepts
                if "source text" in issue:
                    recommendations.append("Improve text preprocessing and concept extraction algorithms")
                elif "keyword" in issue:
                    recommendations.append("Enhance keyword identification and validation")
                elif "coherence" in issue:
                    recommendations.append("Implement better semantic coherence checking")
                elif "attribution" in issue:
                    recommendations.append("Strengthen source attribution mechanisms")
        
        # Score-based recommendations
        overall_integrity = verification_results.get('overall_integrity_score', 0.0)
        if overall_integrity < 0.6:
            recommendations.append("Consider using more conservative concept extraction thresholds")
        
        if failed_count > verified_count:
            recommendations.append("Review document preprocessing - source text quality may be compromised")
        
        if not recommendations:
            recommendations.append("No specific recommendations - verification completed successfully")
        
        return recommendations

# Global verification layer instance
psi_verification = PsiMeshVerificationLayer()

async def verify_concept_extraction_integrity(payload: ParsedPayload) -> Dict[str, Any]:
    """
    Main function for concept extraction verification
    
    Args:
        payload: ParsedPayload with extracted concepts and source content
        
    Returns:
        Comprehensive verification results
    """
    return await psi_verification.verify_concept_extraction(payload)

# Utility functions for integration
def attach_integrity_scores_to_concepts(concepts: List[Dict[str, Any]], 
                                      verification_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Attach integrity scores and verification status to concept objects
    
    Args:
        concepts: Original concept list
        verification_results: Results from verification process
        
    Returns:
        Enhanced concepts with integrity metadata
    """
    enhanced_concepts = []
    
    # Create lookup for verification results
    verified_lookup = {c['concept_name']: c for c in verification_results.get('verified_concepts', [])}
    flagged_lookup = {c['concept_name']: c for c in verification_results.get('flagged_concepts', [])}
    failed_lookup = {c['concept_name']: c for c in verification_results.get('failed_concepts', [])}
    
    for concept in concepts:
        concept_name = concept.get('name', '')
        enhanced_concept = concept.copy()
        
        # Find verification result
        verification_data = None
        verification_status = 'unknown'
        
        if concept_name in verified_lookup:
            verification_data = verified_lookup[concept_name]
            verification_status = 'verified'
        elif concept_name in flagged_lookup:
            verification_data = flagged_lookup[concept_name]
            verification_status = 'flagged'
        elif concept_name in failed_lookup:
            verification_data = failed_lookup[concept_name]
            verification_status = 'failed'
        
        # Attach verification metadata
        enhanced_concept['verification_status'] = verification_status
        
        if verification_data:
            enhanced_concept['integrity_score'] = verification_data.get('integrity_score', 0.0)
            enhanced_concept['verification_checks'] = verification_data.get('verification_checks', {})
            enhanced_concept['source_matches'] = verification_data.get('source_matches', [])
            enhanced_concept['flagged_issues'] = verification_data.get('flagged_issues', [])
        else:
            enhanced_concept['integrity_score'] = 0.0
            enhanced_concept['verification_checks'] = {}
            enhanced_concept['source_matches'] = []
            enhanced_concept['flagged_issues'] = ['Verification not completed']
        
        enhanced_concepts.append(enhanced_concept)
    
    return enhanced_concepts

def filter_concepts_by_integrity(concepts: List[Dict[str, Any]], 
                                min_integrity_score: float = 0.75) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter concepts based on integrity scores
    
    Args:
        concepts: Concepts with integrity scores attached
        min_integrity_score: Minimum integrity score for approval
        
    Returns:
        Tuple of (approved_concepts, rejected_concepts)
    """
    approved = []
    rejected = []
    
    for concept in concepts:
        integrity_score = concept.get('integrity_score', 0.0)
        verification_status = concept.get('verification_status', 'unknown')
        
        if integrity_score >= min_integrity_score and verification_status == 'verified':
            approved.append(concept)
        else:
            rejected.append(concept)
    
    return approved, rejected

def generate_integrity_report(verification_results: Dict[str, Any]) -> str:
    """
    Generate human-readable integrity report
    
    Args:
        verification_results: Results from verification process
        
    Returns:
        Formatted integrity report
    """
    summary = verification_results.get('verification_summary', {})
    
    report = f"""
TORI ψMesh Concept Verification Report
=====================================

Document ID: {verification_results.get('document_id', 'Unknown')}
Verification Timestamp: {verification_results.get('verification_metadata', {}).get('verification_timestamp', 'Unknown')}

SUMMARY
-------
Total Concepts Analyzed: {verification_results.get('total_concepts', 0)}
Verified Concepts: {len(verification_results.get('verified_concepts', []))}
Flagged Concepts: {len(verification_results.get('flagged_concepts', []))}
Failed Concepts: {len(verification_results.get('failed_concepts', []))}

Overall Integrity Score: {verification_results.get('overall_integrity_score', 0.0):.3f}
Verification Rate: {summary.get('verification_rate', 0.0):.1%}
Quality Assessment: {summary.get('quality_assessment', 'Unknown').title()}

RECOMMENDATIONS
--------------
"""
    
    recommendations = verification_results.get('recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec}\n"
    
    # Add details for flagged concepts if any
    flagged_concepts = verification_results.get('flagged_concepts', [])
    if flagged_concepts:
        report += f"\nFLAGGED CONCEPTS DETAILS\n"
        report += f"------------------------\n"
        for concept in flagged_concepts[:5]:  # Show first 5
            report += f"• {concept.get('concept_name', 'Unknown')} (Score: {concept.get('integrity_score', 0.0):.3f})\n"
            issues = concept.get('flagged_issues', [])
            for issue in issues:
                report += f"  - {issue}\n"
    
    return report

# Configuration and constants
VERIFICATION_CONFIG = {
    'default_threshold': 0.75,
    'confidence_threshold': 0.60,
    'source_match_threshold': 0.80,
    'max_concepts_per_document': 50,
    'enable_detailed_logging': True,
    'save_verification_reports': True
}

def update_verification_config(**kwargs):
    """Update verification configuration"""
    global VERIFICATION_CONFIG
    VERIFICATION_CONFIG.update(kwargs)
    
    # Update the verification layer instance
    if 'default_threshold' in kwargs:
        psi_verification.verification_threshold = kwargs['default_threshold']
    if 'confidence_threshold' in kwargs:
        psi_verification.confidence_threshold = kwargs['confidence_threshold']
    if 'source_match_threshold' in kwargs:
        psi_verification.source_match_threshold = kwargs['source_match_threshold']
    
    logger.info(f"ψMesh verification configuration updated: {kwargs}")

def get_verification_statistics() -> Dict[str, Any]:
    """Get current verification layer statistics"""
    return {
        'verification_threshold': psi_verification.verification_threshold,
        'confidence_threshold': psi_verification.confidence_threshold,
        'source_match_threshold': psi_verification.source_match_threshold,
        'check_weights': psi_verification.check_weights,
        'config': VERIFICATION_CONFIG
    }
