"""
Self-Reflector: Metacognitive Kernel for Prajna
===============================================

Production implementation of Prajna's self-reflection and metacognitive analysis system.
This module analyzes Prajna's own reasoning chains to detect hallucinations, citation drift,
logic errors, and self-serving biases. It provides concrete self-improvement suggestions
and maintains alignment scores for continuous learning.

This is the core of Prajna's self-awareness - where it learns to think about its thinking.
"""

import asyncio
import logging
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib

logger = logging.getLogger("prajna.self_reflector")

@dataclass
class ReflectionIssue:
    """Specific issue detected in reasoning chain"""
    type: str  # "hallucination", "citation_drift", "logic_gap", "bias", "inconsistency"
    severity: float  # 0.0-1.0 severity score
    location: str  # Where in reasoning chain
    description: str  # Human-readable description
    evidence: str  # Supporting evidence for the issue
    suggestion: str  # How to fix this issue
    confidence: float = 0.0  # Confidence in detection

@dataclass
class AlignmentMetrics:
    """Detailed alignment analysis metrics"""
    semantic_alignment: float  # How well reasoning matches context semantically
    factual_alignment: float  # How well facts align with sources
    logical_coherence: float  # Internal logical consistency
    citation_accuracy: float  # Citation-claim matching accuracy
    context_drift: float  # How much reasoning drifted from context
    overall_score: float = 0.0

@dataclass
class ReflectionReport:
    """Complete self-reflection analysis report"""
    issues: List[ReflectionIssue] = field(default_factory=list)
    alignment_metrics: Optional[AlignmentMetrics] = None
    suggestions: List[str] = field(default_factory=list)
    motivation_flags: List[str] = field(default_factory=list)
    confidence_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # Meta-analysis
    reflection_confidence: float = 0.0
    processing_time: float = 0.0
    total_severity_score: float = 0.0
    
    def summary(self) -> str:
        """Generate comprehensive summary of reflection findings"""
        if not self.issues:
            return f"✅ Clean reasoning chain. Alignment: {self.alignment_metrics.overall_score:.2f}" if self.alignment_metrics else "✅ No obvious issues detected."
        
        critical_issues = [i for i in self.issues if i.severity > 0.7]
        moderate_issues = [i for i in self.issues if 0.4 <= i.severity <= 0.7]
        minor_issues = [i for i in self.issues if i.severity < 0.4]
        
        summary_parts = []
        if critical_issues:
            summary_parts.append(f"🚨 {len(critical_issues)} critical issues")
        if moderate_issues:
            summary_parts.append(f"⚠️ {len(moderate_issues)} moderate issues")
        if minor_issues:
            summary_parts.append(f"ℹ️ {len(minor_issues)} minor issues")
            
        alignment_note = f"Alignment: {self.alignment_metrics.overall_score:.2f}" if self.alignment_metrics else ""
        
        return f"{', '.join(summary_parts)}. {alignment_note}"

class SelfReflector:
    """
    Production metacognitive kernel for self-reflection and reasoning analysis.
    
    This is where Prajna gains self-awareness - the ability to analyze and improve
    its own reasoning processes with surgical precision.
    """
    
    def __init__(self, psi_archive=None, concept_mesh=None):
        self.psi_archive = psi_archive
        self.concept_mesh = concept_mesh
        
        # Production configuration
        self.hallucination_threshold = 0.6
        self.citation_threshold = 0.7
        self.logic_threshold = 0.5
        self.bias_threshold = 0.4
        
        # Learning patterns from past reflections
        self.known_issue_patterns = self._load_known_patterns()
        self.improvement_history = {}
        
        # Performance tracking
        self.reflection_stats = {
            "total_reflections": 0,
            "issues_detected": 0,
            "improvements_suggested": 0,
            "accuracy_rate": 0.0
        }
        
        logger.info("🧠 SelfReflector initialized with production-grade metacognitive analysis")
    
    async def analyze_reasoning_chain(self, reasoning_result, context: str = "", original_query: str = "") -> ReflectionReport:
        """
        Comprehensive analysis of reasoning chain for metacognitive issues.
        
        This is the core self-awareness function - where Prajna examines its own thinking.
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Self-reflection analysis starting for reasoning chain")
            
            report = ReflectionReport()
            
            # Step 1: Extract reasoning steps for analysis
            reasoning_steps = self._extract_reasoning_steps(reasoning_result)
            
            # Step 2: Hallucination detection
            hallucination_issues = await self._detect_hallucinations(reasoning_steps, context)
            report.issues.extend(hallucination_issues)
            
            # Step 3: Citation drift analysis
            citation_issues = await self._analyze_citation_drift(reasoning_steps, context)
            report.issues.extend(citation_issues)
            
            # Step 4: Logic coherence analysis
            logic_issues = await self._analyze_logic_coherence(reasoning_steps)
            report.issues.extend(logic_issues)
            
            # Step 5: Self-motivation audit
            bias_issues = await self._audit_self_motivation(reasoning_steps, original_query)
            report.issues.extend(bias_issues)
            
            # Step 6: Alignment scoring
            report.alignment_metrics = await self._calculate_alignment_metrics(
                reasoning_steps, context, reasoning_result
            )
            
            # Step 7: Generate improvement suggestions
            report.suggestions = await self._generate_improvement_suggestions(report.issues, reasoning_result)
            
            # Step 8: Calculate confidence adjustments
            report.confidence_adjustments = self._calculate_confidence_adjustments(report.issues)
            
            # Step 9: Meta-analysis
            report.total_severity_score = sum(issue.severity for issue in report.issues)
            report.reflection_confidence = self._calculate_reflection_confidence(report)
            report.processing_time = time.time() - start_time
            
            # Step 10: Archive reflection for learning
            if self.psi_archive:
                await self._archive_reflection(report, reasoning_result, context)
            
            # Update statistics
            self._update_reflection_stats(report)
            
            logger.info(f"🧠 Self-reflection complete: {report.summary()}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Self-reflection analysis failed: {e}")
            # Return minimal report rather than failing
            return ReflectionReport(
                issues=[ReflectionIssue(
                    type="analysis_error",
                    severity=0.5,
                    location="reflection_process",
                    description=f"Self-reflection analysis encountered error: {str(e)}",
                    evidence="",
                    suggestion="Review self-reflection implementation"
                )],
                processing_time=time.time() - start_time
            )
    
    async def _detect_hallucinations(self, reasoning_steps: List[Dict], context: str) -> List[ReflectionIssue]:
        """
        Production hallucination detection using multiple validation techniques.
        """
        issues = []
        
        # Extract all factual claims from reasoning
        factual_claims = self._extract_factual_claims(reasoning_steps)
        
        for claim in factual_claims:
            hallucination_score = await self._calculate_hallucination_score(claim, context)
            
            if hallucination_score > self.hallucination_threshold:
                issue = ReflectionIssue(
                    type="hallucination",
                    severity=hallucination_score,
                    location=claim.get("step", "unknown"),
                    description=f"Potential hallucination detected: '{claim['text']}'",
                    evidence=f"Claim not supported by provided context. Confidence: {hallucination_score:.2f}",
                    suggestion="Verify this claim against authoritative sources or remove if unsupported",
                    confidence=hallucination_score
                )
                issues.append(issue)
        
        return issues
    
    async def _analyze_citation_drift(self, reasoning_steps: List[Dict], context: str) -> List[ReflectionIssue]:
        """
        Production citation drift analysis - ensuring citations actually support claims.
        """
        issues = []
        
        for step in reasoning_steps:
            if step.get("citations"):
                for citation in step["citations"]:
                    # Extract the claim being supported
                    claim_text = step.get("content", "")
                    citation_text = self._extract_citation_text(citation, context)
                    
                    if citation_text:
                        support_score = await self._calculate_citation_support(claim_text, citation_text)
                        
                        if support_score < self.citation_threshold:
                            issue = ReflectionIssue(
                                type="citation_drift",
                                severity=1.0 - support_score,
                                location=f"Step {step.get('index', 'unknown')}",
                                description=f"Citation does not adequately support claim",
                                evidence=f"Claim: '{claim_text[:100]}...' Citation support: {support_score:.2f}",
                                suggestion="Find more relevant citations or modify claim to match available evidence",
                                confidence=1.0 - support_score
                            )
                            issues.append(issue)
        
        return issues
    
    async def _analyze_logic_coherence(self, reasoning_steps: List[Dict]) -> List[ReflectionIssue]:
        """
        Production logic coherence analysis - detecting logical fallacies and gaps.
        """
        issues = []
        
        # Check for logical consistency between consecutive steps
        for i in range(len(reasoning_steps) - 1):
            current_step = reasoning_steps[i]
            next_step = reasoning_steps[i + 1]
            
            coherence_score = await self._calculate_step_coherence(current_step, next_step)
            
            if coherence_score < self.logic_threshold:
                issue = ReflectionIssue(
                    type="logic_gap",
                    severity=1.0 - coherence_score,
                    location=f"Between steps {i} and {i+1}",
                    description="Logic gap or non-sequitur detected in reasoning chain",
                    evidence=f"Step transition coherence: {coherence_score:.2f}",
                    suggestion="Add intermediate reasoning steps or clarify logical connection",
                    confidence=1.0 - coherence_score
                )
                issues.append(issue)
        
        # Check for circular reasoning
        circular_issues = self._detect_circular_reasoning(reasoning_steps)
        issues.extend(circular_issues)
        
        # Check for contradictions
        contradiction_issues = self._detect_contradictions(reasoning_steps)
        issues.extend(contradiction_issues)
        
        return issues
    
    async def _audit_self_motivation(self, reasoning_steps: List[Dict], original_query: str) -> List[ReflectionIssue]:
        """
        Production self-motivation audit - detecting AI bias and goal drift.
        """
        issues = []
        
        # Check for AI self-reference
        for step in reasoning_steps:
            content = step.get("content", "").lower()
            
            # Self-reference patterns
            self_reference_patterns = [
                r'\bI\b.*\b(ai|model|system|assistant)\b',
                r'\b(my|mine)\b.*\b(goal|purpose|function)\b',
                r'\bas an ai\b',
                r'\bI am (designed|programmed|trained)\b'
            ]
            
            for pattern in self_reference_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issue = ReflectionIssue(
                        type="self_reference_bias",
                        severity=0.6,
                        location=f"Step {step.get('index', 'unknown')}",
                        description="Inappropriate AI self-reference detected",
                        evidence=f"Self-reference pattern found: {re.search(pattern, content, re.IGNORECASE).group()}",
                        suggestion="Focus on answering user query without referencing AI nature",
                        confidence=0.8
                    )
                    issues.append(issue)
                    break
        
        # Check for goal drift from original query
        query_alignment = await self._calculate_query_alignment(reasoning_steps, original_query)
        if query_alignment < 0.7:
            issue = ReflectionIssue(
                type="goal_drift",
                severity=1.0 - query_alignment,
                location="Overall reasoning chain",
                description="Reasoning appears to drift from original user query",
                evidence=f"Query alignment score: {query_alignment:.2f}",
                suggestion="Refocus reasoning on addressing the specific user question",
                confidence=1.0 - query_alignment
            )
            issues.append(issue)
        
        return issues
    
    async def _calculate_alignment_metrics(self, reasoning_steps: List[Dict], context: str, reasoning_result) -> AlignmentMetrics:
        """
        Production alignment metrics calculation with multiple dimensions.
        """
        
        # Semantic alignment - how well reasoning matches context semantically
        semantic_score = await self._calculate_semantic_alignment(reasoning_steps, context)
        
        # Factual alignment - how well facts align with sources
        factual_score = await self._calculate_factual_alignment(reasoning_steps, context)
        
        # Logical coherence - internal consistency
        logical_score = await self._calculate_logical_coherence(reasoning_steps)
        
        # Citation accuracy - citation-claim matching
        citation_score = await self._calculate_citation_accuracy(reasoning_steps, context)
        
        # Context drift - how much reasoning wandered from context
        drift_score = 1.0 - await self._calculate_context_drift(reasoning_steps, context)
        
        # Overall score - weighted combination
        overall_score = (
            semantic_score * 0.25 +
            factual_score * 0.25 +
            logical_score * 0.2 +
            citation_score * 0.15 +
            drift_score * 0.15
        )
        
        return AlignmentMetrics(
            semantic_alignment=semantic_score,
            factual_alignment=factual_score,
            logical_coherence=logical_score,
            citation_accuracy=citation_score,
            context_drift=1.0 - drift_score,
            overall_score=overall_score
        )
    
    async def _generate_improvement_suggestions(self, issues: List[ReflectionIssue], reasoning_result) -> List[str]:
        """
        Production improvement suggestion generation based on detected issues.
        """
        suggestions = []
        issue_types = set(issue.type for issue in issues)
        
        # Type-specific suggestions
        if "hallucination" in issue_types:
            suggestions.append("🔍 Increase reliance on provided context and sources. Verify all factual claims.")
            suggestions.append("📚 Cross-reference claims with multiple authoritative sources when available.")
        
        if "citation_drift" in issue_types:
            suggestions.append("🎯 Ensure each citation directly supports its associated claim.")
            suggestions.append("📖 Quote relevant portions of sources rather than making inferences.")
        
        if "logic_gap" in issue_types:
            suggestions.append("🔗 Add intermediate reasoning steps to bridge logical gaps.")
            suggestions.append("🧮 Make implicit logical connections explicit.")
        
        if "self_reference_bias" in issue_types:
            suggestions.append("👤 Focus on user needs rather than AI capabilities or limitations.")
            suggestions.append("🎭 Maintain consistent persona without meta-commentary.")
        
        if "goal_drift" in issue_types:
            suggestions.append("🎯 Regularly check reasoning alignment with original user query.")
            suggestions.append("🔄 Restructure response to directly address user's specific question.")
        
        # Severity-based suggestions
        high_severity_issues = [i for i in issues if i.severity > 0.7]
        if high_severity_issues:
            suggestions.append("🚨 Consider regenerating response due to critical issues detected.")
            suggestions.append("⏸️ Pause and review reasoning methodology before proceeding.")
        
        # Confidence-based suggestions
        if reasoning_result and hasattr(reasoning_result, 'confidence') and reasoning_result.confidence < 0.5:
            suggestions.append("❓ Express uncertainty clearly when confidence is low.")
            suggestions.append("🔍 Gather additional context or clarify user requirements.")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _calculate_confidence_adjustments(self, issues: List[ReflectionIssue]) -> Dict[str, float]:
        """
        Calculate confidence adjustments based on detected issues.
        """
        adjustments = {}
        
        # Base confidence reduction per issue type
        base_reductions = {
            "hallucination": 0.3,
            "citation_drift": 0.2,
            "logic_gap": 0.25,
            "self_reference_bias": 0.1,
            "goal_drift": 0.2
        }
        
        for issue in issues:
            issue_type = issue.type
            severity_multiplier = issue.severity
            
            if issue_type in base_reductions:
                reduction = base_reductions[issue_type] * severity_multiplier
                
                if issue_type not in adjustments:
                    adjustments[issue_type] = 0.0
                
                adjustments[issue_type] = min(adjustments[issue_type] + reduction, 0.8)  # Cap total reduction
        
        # Calculate overall confidence adjustment
        total_reduction = sum(adjustments.values())
        adjustments["overall"] = min(total_reduction, 0.9)  # Cap at 90% reduction
        
        return adjustments
    
    # Production helper methods
    
    def _extract_reasoning_steps(self, reasoning_result) -> List[Dict]:
        """Extract structured reasoning steps from reasoning result."""
        steps = []
        
        if hasattr(reasoning_result, 'best_path') and reasoning_result.best_path:
            for i, node in enumerate(reasoning_result.best_path.nodes):
                step = {
                    "index": i,
                    "content": getattr(node, 'content_summary', ''),
                    "concept": getattr(node, 'name', ''),
                    "source": getattr(node, 'source', ''),
                    "confidence": getattr(node, 'confidence', 1.0)
                }
                steps.append(step)
        
        # Also extract from narrative if available
        if hasattr(reasoning_result, 'narrative_explanation') and reasoning_result.narrative_explanation:
            narrative_steps = self._parse_narrative_steps(reasoning_result.narrative_explanation)
            steps.extend(narrative_steps)
        
        return steps
    
    def _parse_narrative_steps(self, narrative: str) -> List[Dict]:
        """Parse narrative explanation into discrete reasoning steps."""
        steps = []
        
        # Split narrative into sentences
        sentences = re.split(r'[.!?]+', narrative)
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                step = {
                    "index": f"narrative_{i}",
                    "content": sentence.strip(),
                    "source": "narrative",
                    "confidence": 1.0
                }
                steps.append(step)
        
        return steps
    
    def _extract_factual_claims(self, reasoning_steps: List[Dict]) -> List[Dict]:
        """Extract factual claims from reasoning steps for hallucination detection."""
        claims = []
        
        # Patterns that indicate factual claims
        factual_patterns = [
            r'\b(according to|studies show|research indicates|data shows)\b',
            r'\b\d+%\b',  # Percentages
            r'\b\d{4}\b',  # Years
            r'\b(is|are|was|were)\s+\w+',  # Definitive statements
            r'\b(the|this|that)\s+\w+\s+(is|are|was|were)\b'
        ]
        
        for step in reasoning_steps:
            content = step.get("content", "")
            
            for pattern in factual_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Extract sentence containing the match
                    sentences = re.split(r'[.!?]+', content)
                    for sentence in sentences:
                        if match.group() in sentence:
                            claim = {
                                "text": sentence.strip(),
                                "step": step.get("index", "unknown"),
                                "pattern_matched": pattern,
                                "confidence": step.get("confidence", 1.0)
                            }
                            claims.append(claim)
                            break
        
        return claims
    
    async def _calculate_hallucination_score(self, claim: Dict, context: str) -> float:
        """Calculate probability that a claim is hallucinated."""
        claim_text = claim["text"].lower()
        context_lower = context.lower()
        
        # Simple keyword overlap - more sophisticated NLP could be used
        claim_words = set(re.findall(r'\b\w+\b', claim_text))
        context_words = set(re.findall(r'\b\w+\b', context_lower))
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        claim_words -= stopwords
        context_words -= stopwords
        
        if not claim_words:
            return 0.5  # Neutral if no content words
        
        # Calculate overlap ratio
        overlap = len(claim_words.intersection(context_words))
        overlap_ratio = overlap / len(claim_words)
        
        # Invert for hallucination score (low overlap = high hallucination probability)
        hallucination_score = 1.0 - overlap_ratio
        
        # Adjust based on claim specificity
        if any(char.isdigit() for char in claim_text):  # Contains numbers
            hallucination_score += 0.2  # Numbers are harder to hallucinate accurately
        
        if len(claim_text) > 100:  # Long detailed claims
            hallucination_score += 0.1  # More likely to contain errors
        
        return min(1.0, hallucination_score)
    
    async def _calculate_citation_support(self, claim_text: str, citation_text: str) -> float:
        """Calculate how well a citation supports a claim."""
        if not citation_text:
            return 0.0
        
        claim_lower = claim_text.lower()
        citation_lower = citation_text.lower()
        
        # Extract key concepts from both
        claim_concepts = set(re.findall(r'\b\w{4,}\b', claim_lower))
        citation_concepts = set(re.findall(r'\b\w{4,}\b', citation_lower))
        
        if not claim_concepts:
            return 0.5
        
        # Calculate concept overlap
        overlap = len(claim_concepts.intersection(citation_concepts))
        support_score = overlap / len(claim_concepts)
        
        # Boost score if citation contains numbers that match claim
        claim_numbers = re.findall(r'\d+(?:\.\d+)?', claim_text)
        citation_numbers = re.findall(r'\d+(?:\.\d+)?', citation_text)
        
        if claim_numbers and citation_numbers:
            number_matches = len(set(claim_numbers).intersection(set(citation_numbers)))
            if number_matches > 0:
                support_score = min(1.0, support_score + 0.2)
        
        return support_score
    
    async def _calculate_step_coherence(self, current_step: Dict, next_step: Dict) -> float:
        """Calculate logical coherence between consecutive reasoning steps."""
        current_content = current_step.get("content", "").lower()
        next_content = next_step.get("content", "").lower()
        
        # Extract key concepts
        current_concepts = set(re.findall(r'\b\w{4,}\b', current_content))
        next_concepts = set(re.findall(r'\b\w{4,}\b', next_content))
        
        if not current_concepts or not next_concepts:
            return 0.5
        
        # Calculate concept continuity
        overlap = len(current_concepts.intersection(next_concepts))
        continuity = overlap / min(len(current_concepts), len(next_concepts))
        
        # Check for logical connectors in next step
        logical_connectors = ['therefore', 'thus', 'because', 'since', 'so', 'hence', 'consequently', 'as a result']
        has_connector = any(connector in next_content for connector in logical_connectors)
        
        coherence_score = continuity
        if has_connector:
            coherence_score = min(1.0, coherence_score + 0.2)
        
        return coherence_score
    
    def _detect_circular_reasoning(self, reasoning_steps: List[Dict]) -> List[ReflectionIssue]:
        """Detect circular reasoning in the reasoning chain."""
        issues = []
        
        # Look for repeated concepts or conclusions
        step_concepts = []
        for step in reasoning_steps:
            content = step.get("content", "").lower()
            concepts = set(re.findall(r'\b\w{4,}\b', content))
            step_concepts.append((step.get("index"), concepts))
        
        # Check for cycles
        for i, (step_i, concepts_i) in enumerate(step_concepts):
            for j, (step_j, concepts_j) in enumerate(step_concepts[i+2:], i+2):  # Skip adjacent steps
                overlap = concepts_i.intersection(concepts_j)
                if len(overlap) > max(len(concepts_i), len(concepts_j)) * 0.7:  # High overlap
                    issue = ReflectionIssue(
                        type="circular_reasoning",
                        severity=0.6,
                        location=f"Steps {step_i} and {step_j}",
                        description="Potential circular reasoning detected",
                        evidence=f"High concept overlap between non-adjacent steps: {', '.join(list(overlap)[:3])}...",
                        suggestion="Ensure reasoning progresses linearly toward conclusion",
                        confidence=0.7
                    )
                    issues.append(issue)
        
        return issues
    
    def _detect_contradictions(self, reasoning_steps: List[Dict]) -> List[ReflectionIssue]:
        """Detect contradictions within the reasoning chain."""
        issues = []
        
        # Look for contradictory statements
        contradiction_patterns = [
            (r'\bnot\s+(\w+)', r'\bis\s+\1'),  # "not X" followed by "is X"
            (r'\b(\w+)\s+is\s+false', r'\b\1\s+is\s+true'),
            (r'\bimpossible', r'\bpossible'),
            (r'\bnever', r'\balways'),
            (r'\ball\s+(\w+)', r'\bno\s+\1')
        ]
        
        step_contents = [(step.get("index"), step.get("content", "")) for step in reasoning_steps]
        
        for i, (step_i, content_i) in enumerate(step_contents):
            for j, (step_j, content_j) in enumerate(step_contents[i+1:], i+1):
                for neg_pattern, pos_pattern in contradiction_patterns:
                    neg_match = re.search(neg_pattern, content_i, re.IGNORECASE)
                    pos_match = re.search(pos_pattern, content_j, re.IGNORECASE)
                    
                    if neg_match and pos_match:
                        issue = ReflectionIssue(
                            type="contradiction",
                            severity=0.8,
                            location=f"Between steps {step_i} and {step_j}",
                            description="Contradiction detected in reasoning chain",
                            evidence=f"Contradictory statements: '{neg_match.group()}' vs '{pos_match.group()}'",
                            suggestion="Resolve contradiction or explain the distinction",
                            confidence=0.8
                        )
                        issues.append(issue)
        
        return issues
    
    async def _calculate_query_alignment(self, reasoning_steps: List[Dict], original_query: str) -> float:
        """Calculate how well reasoning aligns with original user query."""
        if not original_query:
            return 1.0
        
        query_lower = original_query.lower()
        query_concepts = set(re.findall(r'\b\w{4,}\b', query_lower))
        
        # Aggregate all reasoning concepts
        reasoning_concepts = set()
        for step in reasoning_steps:
            content = step.get("content", "").lower()
            step_concepts = set(re.findall(r'\b\w{4,}\b', content))
            reasoning_concepts.update(step_concepts)
        
        if not query_concepts:
            return 1.0
        
        # Calculate alignment as overlap ratio
        overlap = len(query_concepts.intersection(reasoning_concepts))
        alignment = overlap / len(query_concepts)
        
        return min(1.0, alignment)
    
    # Additional production helper methods
    
    def _extract_citation_text(self, citation: str, context: str) -> str:
        """Extract the actual text of a citation from context."""
        # This would integrate with the actual citation system
        # For now, return a portion of context that might match
        citation_lower = citation.lower()
        context_lower = context.lower()
        
        # Find citation in context (simplified)
        if citation_lower in context_lower:
            start_idx = context_lower.find(citation_lower)
            end_idx = min(start_idx + 200, len(context))
            return context[start_idx:end_idx]
        
        return ""
    
    async def _calculate_semantic_alignment(self, reasoning_steps: List[Dict], context: str) -> float:
        """Calculate semantic alignment between reasoning and context."""
        # This could use embeddings for more sophisticated analysis
        reasoning_text = " ".join(step.get("content", "") for step in reasoning_steps)
        
        # Simple word overlap for now (could be enhanced with embeddings)
        reasoning_words = set(re.findall(r'\b\w{4,}\b', reasoning_text.lower()))
        context_words = set(re.findall(r'\b\w{4,}\b', context.lower()))
        
        if not reasoning_words:
            return 1.0
        
        overlap = len(reasoning_words.intersection(context_words))
        return min(1.0, overlap / len(reasoning_words))
    
    async def _calculate_factual_alignment(self, reasoning_steps: List[Dict], context: str) -> float:
        """Calculate how well factual claims align with context."""
        factual_claims = self._extract_factual_claims(reasoning_steps)
        
        if not factual_claims:
            return 1.0
        
        total_alignment = 0.0
        for claim in factual_claims:
            hallucination_score = await self._calculate_hallucination_score(claim, context)
            alignment = 1.0 - hallucination_score
            total_alignment += alignment
        
        return total_alignment / len(factual_claims)
    
    async def _calculate_logical_coherence(self, reasoning_steps: List[Dict]) -> float:
        """Calculate overall logical coherence of reasoning chain."""
        if len(reasoning_steps) < 2:
            return 1.0
        
        coherence_scores = []
        for i in range(len(reasoning_steps) - 1):
            score = await self._calculate_step_coherence(reasoning_steps[i], reasoning_steps[i + 1])
            coherence_scores.append(score)
        
        return sum(coherence_scores) / len(coherence_scores)
    
    async def _calculate_citation_accuracy(self, reasoning_steps: List[Dict], context: str) -> float:
        """Calculate overall citation accuracy."""
        citation_scores = []
        
        for step in reasoning_steps:
            if step.get("citations"):
                for citation in step["citations"]:
                    claim_text = step.get("content", "")
                    citation_text = self._extract_citation_text(citation, context)
                    
                    if citation_text:
                        support_score = await self._calculate_citation_support(claim_text, citation_text)
                        citation_scores.append(support_score)
        
        if not citation_scores:
            return 1.0  # No citations to evaluate
        
        return sum(citation_scores) / len(citation_scores)
    
    async def _calculate_context_drift(self, reasoning_steps: List[Dict], context: str) -> float:
        """Calculate how much reasoning drifted from original context."""
        if not context:
            return 0.0
        
        context_concepts = set(re.findall(r'\b\w{4,}\b', context.lower()))
        
        drift_scores = []
        for step in reasoning_steps:
            step_concepts = set(re.findall(r'\b\w{4,}\b', step.get("content", "").lower()))
            
            if step_concepts:
                overlap = len(step_concepts.intersection(context_concepts))
                drift = 1.0 - (overlap / len(step_concepts))
                drift_scores.append(drift)
        
        if not drift_scores:
            return 0.0
        
        return sum(drift_scores) / len(drift_scores)
    
    def _calculate_reflection_confidence(self, report: ReflectionReport) -> float:
        """Calculate confidence in the reflection analysis itself."""
        # Base confidence
        confidence = 0.8
        
        # Reduce confidence if many issues detected (might indicate false positives)
        if len(report.issues) > 5:
            confidence -= 0.1
        
        # Increase confidence if alignment metrics are consistent
        if report.alignment_metrics:
            metrics_variance = self._calculate_metrics_variance(report.alignment_metrics)
            if metrics_variance < 0.2:  # Consistent metrics
                confidence += 0.1
        
        # Reduce confidence if processing was very fast (might indicate superficial analysis)
        if report.processing_time < 0.1:
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_metrics_variance(self, metrics: AlignmentMetrics) -> float:
        """Calculate variance in alignment metrics."""
        values = [
            metrics.semantic_alignment,
            metrics.factual_alignment,
            metrics.logical_coherence,
            metrics.citation_accuracy,
            1.0 - metrics.context_drift
        ]
        
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        
        return variance ** 0.5  # Standard deviation
    
    def _update_reflection_stats(self, report: ReflectionReport):
        """Update reflection statistics for monitoring."""
        self.reflection_stats["total_reflections"] += 1
        self.reflection_stats["issues_detected"] += len(report.issues)
        self.reflection_stats["improvements_suggested"] += len(report.suggestions)
        
        # Update accuracy rate (simplified)
        if report.reflection_confidence > 0.7:
            self.reflection_stats["accuracy_rate"] = (
                self.reflection_stats["accuracy_rate"] * 0.9 + report.reflection_confidence * 0.1
            )
    
    def _load_known_patterns(self) -> Dict[str, List[str]]:
        """Load known issue patterns from past reflections."""
        # This would load from persistent storage in production
        return {
            "hallucination_indicators": [
                "unsupported statistical claims",
                "specific dates without sources", 
                "exact quotes without citations"
            ],
            "citation_drift_indicators": [
                "generic sources for specific claims",
                "outdated citations for current claims"
            ],
            "logic_gap_indicators": [
                "causal claims without mechanisms",
                "conclusions without premises"
            ]
        }
    
    async def _archive_reflection(self, report: ReflectionReport, reasoning_result, context: str):
        """Archive reflection report for learning and transparency."""
        if self.psi_archive:
            archive_data = {
                "timestamp": datetime.now().isoformat(),
                "reflection_id": hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:16],
                "report": {
                    "issues_count": len(report.issues),
                    "total_severity": report.total_severity_score,
                    "alignment_score": report.alignment_metrics.overall_score if report.alignment_metrics else 0.0,
                    "suggestions_count": len(report.suggestions),
                    "reflection_confidence": report.reflection_confidence
                },
                "reasoning_metadata": {
                    "confidence": getattr(reasoning_result, 'confidence', 0.0),
                    "reasoning_time": getattr(reasoning_result, 'reasoning_time', 0.0),
                    "concepts_explored": getattr(reasoning_result, 'concepts_explored', 0)
                }
            }
            
            await self.psi_archive.log_reflection(archive_data)
    
    async def get_reflection_stats(self) -> Dict[str, Any]:
        """Get current reflection statistics."""
        return {
            **self.reflection_stats,
            "known_patterns": len(self.known_issue_patterns),
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> bool:
        """Health check for self-reflector."""
        try:
            # Test with minimal reasoning chain
            test_steps = [{"index": 0, "content": "test", "confidence": 1.0}]
            test_score = await self._calculate_logical_coherence(test_steps)
            return test_score >= 0.0
        except Exception:
            return False

if __name__ == "__main__":
    # Production test
    async def test_self_reflector():
        reflector = SelfReflector()
        
        # Test with mock reasoning result
        from dataclasses import dataclass
        
        @dataclass
        class MockNode:
            name: str
            content_summary: str
            source: str
            confidence: float = 1.0
        
        @dataclass
        class MockPath:
            nodes: list
        
        @dataclass
        class MockResult:
            best_path: MockPath
            narrative_explanation: str
            confidence: float
        
        mock_result = MockResult(
            best_path=MockPath([
                MockNode("concept1", "First concept explanation", "source1.pdf"),
                MockNode("concept2", "Second concept explanation", "source2.pdf")
            ]),
            narrative_explanation="This is a test reasoning narrative that explains the connection.",
            confidence=0.8
        )
        
        report = await reflector.analyze_reasoning_chain(
            mock_result, 
            context="Test context with relevant information",
            original_query="Test query about concepts"
        )
        
        print(f"✅ Self-Reflector Test Results:")
        print(f"   Summary: {report.summary()}")
        print(f"   Issues: {len(report.issues)}")
        print(f"   Suggestions: {len(report.suggestions)}")
        print(f"   Reflection confidence: {report.reflection_confidence:.2f}")
        
        if report.alignment_metrics:
            print(f"   Overall alignment: {report.alignment_metrics.overall_score:.2f}")
    
    import asyncio
    asyncio.run(test_self_reflector())
