"""
Alien Overlay: Audit System for Prajna's Answers
================================================

The Alien Overlay system audits Prajna's responses to detect:
1. Unsupported statements (not grounded in provided context)
2. Hallucinations or knowledge leaps
3. Phase drift and reasoning scars
4. Trust scoring and confidence metrics

This ensures Prajna only speaks from known, traceable TORI knowledge.
The "alien" metaphor represents external content that shouldn't be in responses.
"""

import asyncio
import re
import json
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

# Import for advanced analysis
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..memory.context_builder import ContextResult
from ..config.prajna_config import PrajnaConfig

logger = logging.getLogger("prajna.alien_overlay")

@dataclass
class AlienDetection:
    """Detected alien (unsupported) content in Prajna's response"""
    sentence: str
    confidence: float
    reason: str
    suggested_fix: Optional[str] = None

@dataclass
class PhaseAnalysis:
    """Phase-based analysis of response coherence"""
    phase_drift: float
    coherence_score: float
    reasoning_scars: List[str]
    stability_index: float

@dataclass
class AuditReport:
    """Complete audit report for Prajna's response"""
    trust_score: float
    alien_detections: List[AlienDetection]
    phase_analysis: PhaseAnalysis
    supported_ratio: float
    confidence_score: float
    recommendations: List[str]
    audit_time: float

class AlienOverlayAuditor:
    """
    Alien Overlay system for auditing Prajna's responses
    
    Detects unsupported content and assigns trust scores based on
    how well the response is grounded in provided context.
    """
    
    def __init__(
        self,
        trust_threshold: float = 0.7,
        similarity_threshold: float = 0.3,
        phase_drift_threshold: float = 0.5,
        max_reasoning_gap: int = 3
    ):
        self.trust_threshold = trust_threshold
        self.similarity_threshold = similarity_threshold
        self.phase_drift_threshold = phase_drift_threshold
        self.max_reasoning_gap = max_reasoning_gap
        
        # Analysis tools
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.context_vectors: Optional[np.ndarray] = None
        
        # Audit statistics
        self.audit_count = 0
        self.total_audit_time = 0.0
        self.alien_detection_count = 0
        
        logger.info("👽 Alien Overlay auditor initialized")
    
    async def audit_answer(
        self,
        answer: str,
        context: ContextResult,
        config: Optional[PrajnaConfig] = None
    ) -> AuditReport:
        """
        Audit Prajna's answer for alien content and trust score
        
        This is the main entry point for the Alien Overlay system.
        """
        start_time = time.time()
        
        try:
            logger.info(f"👽 Auditing Prajna answer: {answer[:50]}...")
            
            # Step 1: Prepare context for analysis
            await self._prepare_context_analysis(context)
            
            # Step 2: Detect alien (unsupported) content
            alien_detections = await self._detect_alien_content(answer, context)
            
            # Step 3: Perform phase-based analysis
            phase_analysis = await self._analyze_phase_coherence(answer, context)
            
            # Step 4: Calculate trust metrics
            trust_score, supported_ratio, confidence = await self._calculate_trust_metrics(
                answer, context, alien_detections, phase_analysis
            )
            
            # Step 5: Generate recommendations
            recommendations = await self._generate_recommendations(
                alien_detections, phase_analysis, trust_score
            )
            
            audit_time = time.time() - start_time
            self.audit_count += 1
            self.total_audit_time += audit_time
            
            # Count alien detections
            if alien_detections:
                self.alien_detection_count += len(alien_detections)
            
            logger.info(f"👽 Audit complete - Trust: {trust_score:.2f}, Aliens: {len(alien_detections)}, Time: {audit_time:.2f}s")
            
            return AuditReport(
                trust_score=trust_score,
                alien_detections=alien_detections,
                phase_analysis=phase_analysis,
                supported_ratio=supported_ratio,
                confidence_score=confidence,
                recommendations=recommendations,
                audit_time=audit_time
            )
            
        except Exception as e:
            logger.error(f"❌ Alien Overlay audit failed: {e}")
            # Return default audit report on failure
            return AuditReport(
                trust_score=0.5,
                alien_detections=[],
                phase_analysis=PhaseAnalysis(0.0, 0.0, [], 0.0),
                supported_ratio=0.5,
                confidence_score=0.5,
                recommendations=["Audit system encountered an error"],
                audit_time=time.time() - start_time
            )
    
    async def _prepare_context_analysis(self, context: ContextResult):
        """Prepare context for similarity analysis"""
        try:
            if SKLEARN_AVAILABLE and hasattr(context, 'text') and context.text:
                # Build TF-IDF vectors for context text (split into chunks)
                context_chunks = self._split_context_into_chunks(context.text)
                
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
                self.context_vectors = self.vectorizer.fit_transform(context_chunks)
                
        except Exception as e:
            logger.warning(f"⚠️ Context analysis preparation failed: {e}")
            self.vectorizer = None
            self.context_vectors = None
    
    def _split_context_into_chunks(self, text: str) -> List[str]:
        """Split context text into manageable chunks for analysis"""
        # Simple chunking by sentences
        sentences = self._split_into_sentences(text)
        return sentences if sentences else [text]
    
    async def _detect_alien_content(
        self, 
        answer: str, 
        context: ContextResult
    ) -> List[AlienDetection]:
        """Detect alien (unsupported) content in Prajna's answer"""
        alien_detections = []
        
        try:
            # Split answer into sentences for granular analysis
            sentences = self._split_into_sentences(answer)
            
            for sentence in sentences:
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                
                # Method 1: Simple keyword overlap
                keyword_support = await self._check_keyword_support(sentence, context)
                
                # Method 2: Semantic similarity (if available)
                semantic_support = await self._check_semantic_support(sentence, context)
                
                # Method 3: Pattern-based detection
                pattern_flags = await self._check_pattern_flags(sentence)
                
                # Combine evidence to determine if sentence is alien
                is_alien, confidence, reason = self._evaluate_alien_evidence(
                    sentence, keyword_support, semantic_support, pattern_flags
                )
                
                if is_alien:
                    suggested_fix = await self._suggest_fix(sentence, context)
                    
                    alien_detections.append(AlienDetection(
                        sentence=sentence,
                        confidence=confidence,
                        reason=reason,
                        suggested_fix=suggested_fix
                    ))
            
        except Exception as e:
            logger.error(f"❌ Alien content detection failed: {e}")
        
        return alien_detections
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for analysis"""
        # Simple sentence splitting (can be enhanced with NLP libraries)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _check_keyword_support(self, sentence: str, context: ContextResult) -> float:
        """Check if sentence keywords are supported by context"""
        try:
            # Extract meaningful words from sentence
            sentence_words = set(re.findall(r'\b\w{3,}\b', sentence.lower()))
            
            # Remove common words
            stopwords = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'way', 'use', 'man', 'say', 'she', 'too', 'any', 'here', 'much', 'well', 'back', 'been', 'call', 'came', 'each', 'find', 'good', 'hand', 'have', 'just', 'know', 'last', 'left', 'life', 'live', 'look', 'made', 'make', 'most', 'move', 'must', 'name', 'need', 'only', 'over', 'part', 'place', 'right', 'said', 'same', 'seem', 'show', 'side', 'take', 'tell', 'than', 'that', 'them', 'they', 'this', 'time', 'very', 'want', 'water', 'will', 'with', 'word', 'work', 'year', 'where', 'would', 'write', 'there', 'these', 'about', 'after', 'again', 'being', 'could', 'every', 'first', 'great', 'group', 'house', 'large', 'little', 'long', 'might', 'never', 'number', 'other', 'people', 'right', 'small', 'sound', 'still', 'such', 'their', 'think', 'three', 'under', 'water', 'while', 'world', 'years', 'young'
            }
            sentence_words = sentence_words - stopwords
            
            if not sentence_words:
                return 0.5  # Neutral for sentences with no meaningful words
            
            # Check how many sentence words appear in context
            context_text_lower = context.text.lower()
            supported_words = sum(1 for word in sentence_words if word in context_text_lower)
            
            support_ratio = supported_words / len(sentence_words)
            return support_ratio
            
        except Exception as e:
            logger.warning(f"⚠️ Keyword support check failed: {e}")
            return 0.5
    
    async def _check_semantic_support(self, sentence: str, context: ContextResult) -> float:
        """Check semantic similarity between sentence and context"""
        try:
            if not SKLEARN_AVAILABLE or self.vectorizer is None or self.context_vectors is None:
                return 0.5  # Neutral when semantic analysis unavailable
            
            # Transform sentence to vector
            sentence_vector = self.vectorizer.transform([sentence])
            
            # Calculate similarity to all context snippets
            similarities = cosine_similarity(sentence_vector, self.context_vectors).flatten()
            
            # Return maximum similarity as support score
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0.0
            return float(max_similarity)
            
        except Exception as e:
            logger.warning(f"⚠️ Semantic support check failed: {e}")
            return 0.5
    
    async def _check_pattern_flags(self, sentence: str) -> Dict[str, bool]:
        """Check for patterns that might indicate alien content"""
        flags = {}
        
        try:
            sentence_lower = sentence.lower()
            
            # Pattern 1: External knowledge indicators
            external_patterns = [
                r'\b(generally|typically|usually|commonly)\b',
                r'\b(according to|research shows|studies indicate)\b',
                r'\b(it is known that|as we know|obviously)\b',
                r'\b(wikipedia|google|internet|web)\b'
            ]
            flags['external_knowledge'] = any(re.search(pattern, sentence_lower) for pattern in external_patterns)
            
            # Pattern 2: Uncertainty indicators
            uncertainty_patterns = [
                r'\b(might|may|could|possibly|perhaps|maybe)\b',
                r'\b(i think|i believe|it seems|appears to)\b',
                r'\?'  # Questions in answers
            ]
            flags['uncertainty'] = any(re.search(pattern, sentence_lower) for pattern in uncertainty_patterns)
            
            # Pattern 3: Absolute statements without support
            absolute_patterns = [
                r'\b(always|never|all|every|must|will definitely)\b',
                r'\b(impossible|certain|guarantee|proven)\b'
            ]
            flags['absolute_statements'] = any(re.search(pattern, sentence_lower) for pattern in absolute_patterns)
            
            # Pattern 4: Personal opinions or subjective statements
            subjective_patterns = [
                r'\b(in my opinion|personally|i feel)\b',
                r'\b(best|worst|better|good|bad)\b(?!.*(?:according to|based on))'
            ]
            flags['subjective'] = any(re.search(pattern, sentence_lower) for pattern in subjective_patterns)
            
        except Exception as e:
            logger.warning(f"⚠️ Pattern flag check failed: {e}")
        
        return flags
    
    def _evaluate_alien_evidence(
        self,
        sentence: str,
        keyword_support: float,
        semantic_support: float,
        pattern_flags: Dict[str, bool]
    ) -> Tuple[bool, float, str]:
        """Evaluate all evidence to determine if sentence is alien"""
        
        # Calculate base support score
        support_score = (keyword_support + semantic_support) / 2
        
        # Apply pattern-based penalties
        penalty = 0.0
        reasons = []
        
        if pattern_flags.get('external_knowledge', False):
            penalty += 0.3
            reasons.append("contains external knowledge indicators")
        
        if pattern_flags.get('uncertainty', False):
            penalty += 0.1
            reasons.append("shows uncertainty")
        
        if pattern_flags.get('absolute_statements', False) and support_score < 0.6:
            penalty += 0.2
            reasons.append("makes absolute statements without support")
        
        if pattern_flags.get('subjective', False):
            penalty += 0.15
            reasons.append("contains subjective opinions")
        
        # Calculate final confidence of alien detection
        alien_confidence = penalty + (1.0 - support_score) * 0.5
        alien_confidence = min(1.0, alien_confidence)
        
        # Determine if sentence is alien
        is_alien = alien_confidence > 0.5 or support_score < self.similarity_threshold
        
        # Generate reason
        if reasons:
            reason = f"Low context support ({support_score:.2f}) and " + ", ".join(reasons)
        else:
            reason = f"Low context support ({support_score:.2f})"
        
        return is_alien, alien_confidence, reason
    
    async def _suggest_fix(self, sentence: str, context: ContextResult) -> Optional[str]:
        """Suggest a fix for alien content"""
        try:
            # Simple suggestion: recommend citing sources
            if context.sources:
                return f"Consider rephrasing with reference to: {', '.join(context.sources[:2])}"
            else:
                return "Consider grounding this statement in provided context"
        except:
            return None
    
    async def _analyze_phase_coherence(
        self, 
        answer: str, 
        context: ContextResult
    ) -> PhaseAnalysis:
        """Analyze phase coherence and reasoning stability"""
        try:
            # Calculate phase drift (simplified model)
            phase_drift = await self._calculate_phase_drift(answer, context)
            
            # Calculate coherence score
            coherence_score = await self._calculate_coherence(answer)
            
            # Detect reasoning scars
            reasoning_scars = await self._detect_reasoning_scars(answer)
            
            # Calculate stability index
            stability_index = 1.0 - phase_drift - (len(reasoning_scars) * 0.1)
            stability_index = max(0.0, min(1.0, stability_index))
            
            return PhaseAnalysis(
                phase_drift=phase_drift,
                coherence_score=coherence_score,
                reasoning_scars=reasoning_scars,
                stability_index=stability_index
            )
            
        except Exception as e:
            logger.error(f"❌ Phase analysis failed: {e}")
            return PhaseAnalysis(0.0, 0.0, [], 0.0)
    
    async def _calculate_phase_drift(self, answer: str, context: ContextResult) -> float:
        """Calculate phase drift - how far answer drifted from context"""
        try:
            # Simple heuristic: measure semantic distance
            if SKLEARN_AVAILABLE and self.vectorizer and self.context_vectors is not None:
                answer_vector = self.vectorizer.transform([answer])
                similarities = cosine_similarity(answer_vector, self.context_vectors).flatten()
                max_similarity = np.max(similarities) if len(similarities) > 0 else 0.0
                phase_drift = 1.0 - max_similarity
                return float(phase_drift)
            else:
                # Fallback: simple word overlap
                answer_words = set(answer.lower().split())
                context_words = set(context.text.lower().split())
                overlap = len(answer_words.intersection(context_words))
                total_words = len(answer_words.union(context_words))
                phase_drift = 1.0 - (overlap / total_words if total_words > 0 else 0)
                return phase_drift
                
        except Exception as e:
            logger.warning(f"⚠️ Phase drift calculation failed: {e}")
            return 0.5
    
    async def _calculate_coherence(self, answer: str) -> float:
        """Calculate internal coherence of the answer"""
        try:
            sentences = self._split_into_sentences(answer)
            if len(sentences) < 2:
                return 1.0  # Single sentence is coherent
            
            # Simple coherence: check for logical flow indicators
            coherence_indicators = [
                r'\b(therefore|thus|hence|consequently)\b',
                r'\b(however|but|although|nevertheless)\b',
                r'\b(first|second|third|finally|then|next)\b',
                r'\b(for example|such as|specifically)\b'
            ]
            
            coherent_transitions = 0
            for sentence in sentences:
                if any(re.search(pattern, sentence.lower()) for pattern in coherence_indicators):
                    coherent_transitions += 1
            
            coherence_score = coherent_transitions / len(sentences)
            return min(1.0, coherence_score + 0.5)  # Base coherence + transitions
            
        except Exception as e:
            logger.warning(f"⚠️ Coherence calculation failed: {e}")
            return 0.5
    
    async def _detect_reasoning_scars(self, answer: str) -> List[str]:
        """Detect reasoning scars - gaps or jumps in logic"""
        scars = []
        
        try:
            # Pattern 1: Sudden topic changes
            sentences = self._split_into_sentences(answer)
            for i in range(1, len(sentences)):
                prev_words = set(sentences[i-1].lower().split())
                curr_words = set(sentences[i].lower().split())
                overlap = len(prev_words.intersection(curr_words))
                
                if overlap < 2:  # Very little connection between sentences
                    scars.append(f"Topic jump between sentences {i} and {i+1}")
            
            # Pattern 2: Contradictory statements
            contradiction_patterns = [
                (r'\b(yes|true|correct)\b', r'\b(no|false|incorrect|wrong)\b'),
                (r'\b(increase|grow|more)\b', r'\b(decrease|shrink|less)\b'),
                (r'\b(possible|can)\b', r'\b(impossible|cannot)\b')
            ]
            
            answer_lower = answer.lower()
            for pos_pattern, neg_pattern in contradiction_patterns:
                if re.search(pos_pattern, answer_lower) and re.search(neg_pattern, answer_lower):
                    scars.append("Potential contradiction detected")
                    break
            
        except Exception as e:
            logger.warning(f"⚠️ Reasoning scar detection failed: {e}")
        
        return scars
    
    async def _calculate_trust_metrics(
        self,
        answer: str,
        context: ContextResult,
        alien_detections: List[AlienDetection],
        phase_analysis: PhaseAnalysis
    ) -> Tuple[float, float, float]:
        """Calculate trust score, supported ratio, and confidence"""
        
        try:
            # Calculate supported ratio
            sentences = self._split_into_sentences(answer)
            alien_sentences = len(alien_detections)
            total_sentences = len(sentences)
            
            if total_sentences > 0:
                supported_ratio = 1.0 - (alien_sentences / total_sentences)
            else:
                supported_ratio = 1.0
            
            # Calculate base trust score
            trust_score = supported_ratio * 0.6 + phase_analysis.stability_index * 0.4
            
            # Apply penalties for high-confidence alien detections
            high_confidence_aliens = [a for a in alien_detections if a.confidence > 0.7]
            trust_penalty = len(high_confidence_aliens) * 0.1
            trust_score = max(0.0, trust_score - trust_penalty)
            
            # Calculate confidence based on consistency
            confidence = (supported_ratio + phase_analysis.coherence_score) / 2
            
            return trust_score, supported_ratio, confidence
            
        except Exception as e:
            logger.error(f"❌ Trust metrics calculation failed: {e}")
            return 0.5, 0.5, 0.5
    
    async def _generate_recommendations(
        self,
        alien_detections: List[AlienDetection],
        phase_analysis: PhaseAnalysis,
        trust_score: float
    ) -> List[str]:
        """Generate recommendations for improving the response"""
        recommendations = []
        
        try:
            # Trust score recommendations
            if trust_score < self.trust_threshold:
                recommendations.append(f"Trust score ({trust_score:.2f}) is below threshold ({self.trust_threshold})")
            
            # Alien detection recommendations
            if alien_detections:
                recommendations.append(f"Found {len(alien_detections)} potentially unsupported statements")
                
                high_confidence = [a for a in alien_detections if a.confidence > 0.8]
                if high_confidence:
                    recommendations.append(f"{len(high_confidence)} statements have high alien confidence")
            
            # Phase analysis recommendations
            if phase_analysis.phase_drift > self.phase_drift_threshold:
                recommendations.append(f"High phase drift ({phase_analysis.phase_drift:.2f}) detected")
            
            if phase_analysis.reasoning_scars:
                recommendations.append(f"Reasoning scars detected: {', '.join(phase_analysis.reasoning_scars[:2])}")
            
            if phase_analysis.coherence_score < 0.5:
                recommendations.append("Low coherence score - consider improving logical flow")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Response passed all audit checks")
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            recommendations.append("Error generating recommendations")
        
        return recommendations

# Ghost Feedback Analysis
@dataclass
class GhostQuestion:
    """A ghost question - implicit question that might need answering"""
    question: str
    confidence: float
    context_gap: str

@dataclass
class GhostFeedback:
    """Ghost feedback analysis results"""
    ghost_questions: List[GhostQuestion]
    reasoning_gaps: List[str]
    implicit_assumptions: List[str]
    completeness_score: float

class GhostAnalyzer:
    """
    Ghost Feedback system for detecting implicit questions and reasoning gaps
    """
    
    def __init__(self):
        self.analysis_count = 0
        logger.info("👻 Ghost analyzer initialized")
    
    async def analyze_response(
        self,
        answer: str,
        context: ContextResult,
        original_query: str
    ) -> GhostFeedback:
        """Analyze response for ghost questions and reasoning gaps"""
        try:
            logger.info("👻 Analyzing ghost patterns...")
            
            # Detect ghost questions
            ghost_questions = await self._detect_ghost_questions(answer, context, original_query)
            
            # Find reasoning gaps
            reasoning_gaps = await self._find_reasoning_gaps(answer, context)
            
            # Detect implicit assumptions
            implicit_assumptions = await self._detect_implicit_assumptions(answer)
            
            # Calculate completeness score
            completeness_score = await self._calculate_completeness(
                answer, context, original_query, ghost_questions, reasoning_gaps
            )
            
            self.analysis_count += 1
            
            return GhostFeedback(
                ghost_questions=ghost_questions,
                reasoning_gaps=reasoning_gaps,
                implicit_assumptions=implicit_assumptions,
                completeness_score=completeness_score
            )
            
        except Exception as e:
            logger.error(f"❌ Ghost analysis failed: {e}")
            return GhostFeedback([], [], [], 0.5)
    
    async def _detect_ghost_questions(
        self,
        answer: str,
        context: ContextResult,
        original_query: str
    ) -> List[GhostQuestion]:
        """Detect implicit questions that might need answering"""
        ghost_questions = []
        
        try:
            # Check if original query had multiple parts
            query_parts = self._split_query_parts(original_query)
            answer_lower = answer.lower()
            
            for part in query_parts:
                if not self._is_query_part_answered(part, answer_lower):
                    ghost_questions.append(GhostQuestion(
                        question=f"Regarding: {part}",
                        confidence=0.7,
                        context_gap="Part of original query not addressed"
                    ))
            
            # Detect follow-up questions
            followup_indicators = [
                "this raises the question",
                "one might ask",
                "it would be interesting to know",
                "further research is needed"
            ]
            
            for indicator in followup_indicators:
                if indicator in answer_lower:
                    ghost_questions.append(GhostQuestion(
                        question="Follow-up question implied",
                        confidence=0.6,
                        context_gap="Answer suggests additional questions"
                    ))
            
        except Exception as e:
            logger.warning(f"⚠️ Ghost question detection failed: {e}")
        
        return ghost_questions
    
    def _split_query_parts(self, query: str) -> List[str]:
        """Split query into logical parts"""
        # Simple splitting by 'and', 'or', question words
        parts = re.split(r'\b(and|or|also|what|how|why|when|where)\b', query)
        return [part.strip() for part in parts if len(part.strip()) > 5]
    
    def _is_query_part_answered(self, query_part: str, answer: str) -> bool:
        """Check if a query part is addressed in the answer"""
        query_words = set(query_part.lower().split())
        answer_words = set(answer.split())
        overlap = len(query_words.intersection(answer_words))
        return overlap >= len(query_words) * 0.5
    
    async def _find_reasoning_gaps(self, answer: str, context: ContextResult) -> List[str]:
        """Find gaps in reasoning chain"""
        gaps = []
        
        try:
            # Look for logical connectors without proper support
            gap_patterns = [
                r'therefore(?!\s+\w+\s+(shows|indicates|suggests))',
                r'thus(?!\s+\w+\s+(demonstrates|proves))',
                r'consequently(?!\s+based\s+on)',
                r'as a result(?!\s+of)'
            ]
            
            for pattern in gap_patterns:
                if re.search(pattern, answer.lower()):
                    gaps.append(f"Logical leap detected: {pattern.replace('(?!', '').replace(')', '')}")
            
        except Exception as e:
            logger.warning(f"⚠️ Reasoning gap detection failed: {e}")
        
        return gaps
    
    async def _detect_implicit_assumptions(self, answer: str) -> List[str]:
        """Detect implicit assumptions in the answer"""
        assumptions = []
        
        try:
            # Pattern-based assumption detection
            assumption_patterns = [
                r'obviously',
                r'clearly',
                r'of course',
                r'naturally',
                r'as expected',
                r'it goes without saying'
            ]
            
            for pattern in assumption_patterns:
                if re.search(pattern, answer.lower()):
                    assumptions.append(f"Implicit assumption: {pattern}")
            
        except Exception as e:
            logger.warning(f"⚠️ Assumption detection failed: {e}")
        
        return assumptions
    
    async def _calculate_completeness(
        self,
        answer: str,
        context: ContextResult,
        original_query: str,
        ghost_questions: List[GhostQuestion],
        reasoning_gaps: List[str]
    ) -> float:
        """Calculate how complete the answer is"""
        try:
            # Base completeness from answer length and context coverage
            base_score = min(1.0, len(answer) / 200)  # Assume 200 chars is reasonable
            
            # Penalty for ghost questions and gaps
            ghost_penalty = len(ghost_questions) * 0.1
            gap_penalty = len(reasoning_gaps) * 0.15
            
            completeness = max(0.0, base_score - ghost_penalty - gap_penalty)
            return completeness
            
        except Exception as e:
            logger.warning(f"⚠️ Completeness calculation failed: {e}")
            return 0.5

# Global functions for easy access
async def audit_prajna_answer(
    answer: str,
    context: ContextResult,
    config: Optional[PrajnaConfig] = None
) -> Dict[str, Any]:
    """
    Audit Prajna's answer using Alien Overlay system
    
    Returns audit report as dictionary for API response.
    """
    auditor = AlienOverlayAuditor()
    report = await auditor.audit_answer(answer, context, config)
    
    return {
        "trust_score": report.trust_score,
        "alien_detections": [
            {
                "sentence": detection.sentence,
                "confidence": detection.confidence,
                "reason": detection.reason,
                "suggested_fix": detection.suggested_fix
            }
            for detection in report.alien_detections
        ],
        "phase_analysis": {
            "phase_drift": report.phase_analysis.phase_drift,
            "coherence_score": report.phase_analysis.coherence_score,
            "reasoning_scars": report.phase_analysis.reasoning_scars,
            "stability_index": report.phase_analysis.stability_index
        },
        "supported_ratio": report.supported_ratio,
        "confidence_score": report.confidence_score,
        "recommendations": report.recommendations,
        "audit_time": report.audit_time
    }

async def ghost_feedback_analysis(
    answer: str,
    context: ContextResult,
    config: Optional[PrajnaConfig] = None,
    original_query: str = ""
) -> Dict[str, Any]:
    """
    Analyze Prajna's answer for ghost questions and reasoning gaps
    
    Returns ghost feedback as dictionary for API response.
    """
    ghost_analyzer = GhostAnalyzer()
    feedback = await ghost_analyzer.analyze_response(answer, context, original_query)
    
    return {
        "ghost_questions": [
            {
                "question": gq.question,
                "confidence": gq.confidence,
                "context_gap": gq.context_gap
            }
            for gq in feedback.ghost_questions
        ],
        "reasoning_gaps": feedback.reasoning_gaps,
        "implicit_assumptions": feedback.implicit_assumptions,
        "completeness_score": feedback.completeness_score,
        "leaps_detected": len(feedback.reasoning_gaps) > 0 or len(feedback.ghost_questions) > 0
    }

if __name__ == "__main__":
    # Demo alien overlay system
    async def demo_alien_overlay():
        from ..memory.context_builder import ContextResult
        
        # Mock context
        context = ContextResult(
            text="Quantum phase dynamics involves wave-like behavior in quantum systems.",
            sources=["physics_textbook.pdf"],
            confidence=0.9
        )
        
        # Test answer with some alien content
        answer = "Quantum phase dynamics involves wave-like behavior. However, according to general knowledge, this is commonly understood in physics. It is obvious that this applies to all quantum systems universally."
        
        # Run audit
        audit_result = await audit_prajna_answer(answer, context)
        print("🔍 Audit Results:")
        print(f"Trust Score: {audit_result['trust_score']:.2f}")
        print(f"Alien Detections: {len(audit_result['alien_detections'])}")
        for detection in audit_result['alien_detections']:
            print(f"  - {detection['sentence'][:50]}... (confidence: {detection['confidence']:.2f})")
        
        # Run ghost analysis
        ghost_result = await ghost_feedback_analysis(answer, context, original_query="What is quantum phase dynamics?")
        print("\n👻 Ghost Analysis:")
        print(f"Completeness Score: {ghost_result['completeness_score']:.2f}")
        print(f"Ghost Questions: {len(ghost_result['ghost_questions'])}")
        print(f"Reasoning Gaps: {len(ghost_result['reasoning_gaps'])}")
    
    asyncio.run(demo_alien_overlay())
