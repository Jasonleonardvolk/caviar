"""
Concept Synthesizer - ADVANCED CONCEPT FUSION ENGINE
====================================================

Advanced concept synthesis using semantic analysis, graph theory, and 
creative combination strategies for generating meaningful higher-order concepts.
"""

import json
import math
import random
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
import networkx as nx
import logging

logger = logging.getLogger("prajna.evolution.concept_synthesizer")

class ConceptSynthesizer:
    """
    Advanced concept synthesis engine for creating meaningful concept fusions.
    Uses multiple strategies including semantic analysis, graph topology, and creative algorithms.
    """
    
    def __init__(self, concept_graph: nx.Graph = None):
        self.graph = concept_graph or nx.Graph()
        self.synthesis_history = []
        self.semantic_cache = {}
        
        # Synthesis parameters
        self.creativity_factor = 0.7
        self.semantic_threshold = 0.3
        self.coherence_weight = 0.6
        self.novelty_weight = 0.4
        
        # Knowledge domains for intelligent synthesis
        self.domain_keywords = {
            'cognitive': ['cognition', 'thinking', 'reasoning', 'memory', 'attention', 'consciousness'],
            'neural': ['neural', 'neuron', 'synapse', 'brain', 'cortex', 'network'],
            'computational': ['algorithm', 'computation', 'processing', 'model', 'system', 'architecture'],
            'quantum': ['quantum', 'wave', 'field', 'coherence', 'entanglement', 'superposition'],
            'dynamic': ['dynamic', 'oscillation', 'phase', 'synchrony', 'rhythm', 'frequency'],
            'mathematical': ['equation', 'function', 'matrix', 'vector', 'topology', 'geometry']
        }
        
        logger.info("🧬 Initialized Advanced Concept Synthesizer")
    
    def analyze_semantic_structure(self, concept: str) -> Dict:
        """Analyze the semantic structure of a concept"""
        if concept in self.semantic_cache:
            return self.semantic_cache[concept]
        
        # Tokenize and analyze
        tokens = self._tokenize_concept(concept)
        
        analysis = {
            'tokens': tokens,
            'length': len(tokens),
            'domains': self._identify_domains(tokens),
            'complexity': self._calculate_complexity(tokens),
            'abstractness': self._calculate_abstractness(tokens),
            'technical_level': self._assess_technical_level(tokens)
        }
        
        self.semantic_cache[concept] = analysis
        return analysis
    
    def _tokenize_concept(self, concept: str) -> List[str]:
        """Advanced tokenization preserving meaningful units"""
        # Normalize
        concept = concept.lower().strip()
        
        # Split on common separators but preserve compound terms
        tokens = re.split(r'[-_\s]+', concept)
        
        # Filter out empty and very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        
        return tokens
    
    def _identify_domains(self, tokens: List[str]) -> List[str]:
        """Identify knowledge domains based on tokens"""
        domains = []
        
        for domain, keywords in self.domain_keywords.items():
            for token in tokens:
                if any(keyword in token or token in keyword for keyword in keywords):
                    domains.append(domain)
                    break
        
        return list(set(domains))
    
    def _calculate_complexity(self, tokens: List[str]) -> float:
        """Calculate concept complexity based on token analysis"""
        # Factors: length, technical terms, compound structure
        length_factor = min(1.0, len(tokens) / 4.0)
        
        # Technical term detection
        technical_indicators = ['tion', 'sion', 'ment', 'ness', 'ity', 'ism', 'ology', 'graphy']
        technical_count = sum(1 for token in tokens 
                            for indicator in technical_indicators 
                            if token.endswith(indicator))
        technical_factor = min(1.0, technical_count / len(tokens))
        
        return (length_factor + technical_factor) / 2.0
    
    def _calculate_abstractness(self, tokens: List[str]) -> float:
        """Calculate how abstract vs concrete a concept is"""
        abstract_indicators = ['model', 'system', 'framework', 'theory', 'concept', 'approach']
        concrete_indicators = ['device', 'tool', 'machine', 'hardware', 'implementation']
        
        abstract_score = sum(1 for token in tokens if token in abstract_indicators)
        concrete_score = sum(1 for token in tokens if token in concrete_indicators)
        
        if abstract_score + concrete_score == 0:
            return 0.5  # Neutral
        
        return abstract_score / (abstract_score + concrete_score)
    
    def _assess_technical_level(self, tokens: List[str]) -> str:
        """Assess technical sophistication level"""
        technical_tokens = sum(1 for token in tokens if len(token) > 6)
        domain_count = len(self._identify_domains(tokens))
        
        if technical_tokens > 2 and domain_count > 1:
            return 'advanced'
        elif technical_tokens > 0 or domain_count > 0:
            return 'intermediate'
        else:
            return 'basic'
    
    async def synthesize_semantic_fusion(self, concepts: List[str], target_domains: List[str] = None) -> List[Dict]:
        """
        Create semantic fusions that preserve meaning while creating novel combinations.
        """
        logger.info(f"🧬 Synthesizing semantic fusion from {len(concepts)} concepts...")
        
        fusions = []
        
        # Analyze all input concepts
        analyses = {concept: self.analyze_semantic_structure(concept) for concept in concepts}
        
        # Find meaningful combinations
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                fusion = await self._create_semantic_fusion(concept1, concept2, analyses, target_domains)
                if fusion:
                    fusions.append(fusion)
        
        # Sort by synthesis quality
        fusions.sort(key=lambda x: x.get('synthesis_score', 0), reverse=True)
        
        logger.info(f"✅ Generated {len(fusions)} semantic fusions")
        return fusions[:10]  # Return top 10
    
    async def _create_semantic_fusion(self, concept1: str, concept2: str, analyses: Dict, target_domains: List[str]) -> Optional[Dict]:
        """Create a meaningful semantic fusion between two concepts"""
        
        analysis1 = analyses[concept1]
        analysis2 = analyses[concept2]
        
        # Check domain compatibility
        domains1 = set(analysis1['domains'])
        domains2 = set(analysis2['domains'])
        shared_domains = domains1 & domains2
        complementary_domains = domains1 | domains2
        
        # Skip if concepts are too similar or incompatible
        if len(shared_domains) == len(complementary_domains):  # Too similar
            return None
        
        if not shared_domains and len(complementary_domains) > 4:  # Too different
            return None
        
        # Create fusion strategies
        fusion_strategies = []
        
        # Strategy 1: Domain bridge
        if shared_domains:
            bridge_name = self._create_domain_bridge(concept1, concept2, shared_domains)
            fusion_strategies.append(('domain_bridge', bridge_name))
        
        # Strategy 2: Functional combination
        functional_name = self._create_functional_combination(concept1, concept2, analysis1, analysis2)
        fusion_strategies.append(('functional', functional_name))
        
        # Strategy 3: Hierarchical abstraction
        if analysis1['abstractness'] != analysis2['abstractness']:
            abstract_name = self._create_hierarchical_abstraction(concept1, concept2, analysis1, analysis2)
            fusion_strategies.append(('hierarchical', abstract_name))
        
        # Select best strategy
        best_strategy, fusion_name = self._select_best_fusion_strategy(fusion_strategies, target_domains)
        
        if not fusion_name:
            return None
        
        # Calculate synthesis metrics
        synthesis_score = self._calculate_synthesis_score(concept1, concept2, fusion_name, analyses)
        coherence_score = self._calculate_coherence_score(fusion_name, [concept1, concept2])
        novelty_score = self._calculate_novelty_score(fusion_name)
        
        return {
            'canonical_name': self._normalize_name(fusion_name),
            'fusion_name': fusion_name,
            'parents': [concept1, concept2],
            'synthesis_strategy': best_strategy,
            'synthesis_score': synthesis_score,
            'coherence_score': coherence_score,
            'novelty_score': novelty_score,
            'domains': list(complementary_domains),
            'technical_level': max(analysis1['technical_level'], analysis2['technical_level']),
            'concept_hash': self._generate_hash(fusion_name),
            'creation_timestamp': datetime.now().isoformat()
        }
    
    def _create_domain_bridge(self, concept1: str, concept2: str, shared_domains: Set[str]) -> str:
        """Create fusion name based on shared domain concepts"""
        # Extract domain-specific tokens
        tokens1 = self._tokenize_concept(concept1)
        tokens2 = self._tokenize_concept(concept2)
        
        # Find bridging terms
        bridge_terms = []
        for domain in shared_domains:
            domain_keywords = self.domain_keywords.get(domain, [])
            for token in tokens1 + tokens2:
                if any(keyword in token for keyword in domain_keywords):
                    bridge_terms.append(token)
        
        # Create bridge name
        unique_terms = list(set(bridge_terms))[:3]  # Max 3 terms
        
        if len(unique_terms) >= 2:
            return f"{unique_terms[0]}-{unique_terms[1]}-bridge"
        else:
            return f"{concept1.split()[0]}-{concept2.split()[0]}-synthesis"
    
    def _create_functional_combination(self, concept1: str, concept2: str, analysis1: Dict, analysis2: Dict) -> str:
        """Create fusion based on functional relationships"""
        tokens1 = analysis1['tokens']
        tokens2 = analysis2['tokens']
        
        # Identify functional terms
        functional_terms = []
        
        # Look for action/process terms
        action_indicators = ['ing', 'tion', 'sion', 'ment']
        for token in tokens1 + tokens2:
            if any(token.endswith(indicator) for indicator in action_indicators):
                functional_terms.append(token)
        
        # Look for system/structure terms
        structure_terms = ['system', 'network', 'model', 'framework', 'architecture']
        for token in tokens1 + tokens2:
            if token in structure_terms:
                functional_terms.append(token)
        
        # Create functional combination
        if functional_terms:
            primary_function = functional_terms[0]
            return f"adaptive-{primary_function}"
        else:
            # Fallback: create process-oriented name
            return f"{tokens1[0]}-{tokens2[0]}-process"
    
    def _create_hierarchical_abstraction(self, concept1: str, concept2: str, analysis1: Dict, analysis2: Dict) -> str:
        """Create hierarchical abstraction from concepts at different levels"""
        
        # Determine which is more abstract
        if analysis1['abstractness'] > analysis2['abstractness']:
            abstract_concept = concept1
            concrete_concept = concept2
        else:
            abstract_concept = concept2
            concrete_concept = concept1
        
        # Extract key terms
        abstract_tokens = self._tokenize_concept(abstract_concept)
        concrete_tokens = self._tokenize_concept(concrete_concept)
        
        # Create abstraction
        if 'model' in abstract_tokens or 'system' in abstract_tokens:
            return f"{concrete_tokens[0]}-guided-{abstract_tokens[0]}"
        else:
            return f"meta-{concrete_tokens[0]}-{abstract_tokens[0]}"
    
    def _select_best_fusion_strategy(self, strategies: List[Tuple[str, str]], target_domains: List[str]) -> Tuple[str, str]:
        """Select the best fusion strategy based on quality metrics"""
        
        scored_strategies = []
        
        for strategy_type, fusion_name in strategies:
            if not fusion_name:
                continue
                
            # Score based on multiple factors
            score = 0.0
            
            # Length appropriateness
            if 10 <= len(fusion_name) <= 30:
                score += 0.3
            
            # Meaningful structure (has hyphens, reasonable word count)
            if '-' in fusion_name and 2 <= len(fusion_name.split('-')) <= 4:
                score += 0.3
            
            # Domain alignment
            if target_domains:
                fusion_domains = self._identify_domains(self._tokenize_concept(fusion_name))
                domain_overlap = len(set(fusion_domains) & set(target_domains))
                score += (domain_overlap / len(target_domains)) * 0.4
            else:
                score += 0.2  # Neutral score if no target domains
            
            scored_strategies.append((score, strategy_type, fusion_name))
        
        if not scored_strategies:
            return ('fallback', None)
        
        # Return highest scoring strategy
        scored_strategies.sort(reverse=True)
        _, best_strategy, best_name = scored_strategies[0]
        
        return (best_strategy, best_name)
    
    def _calculate_synthesis_score(self, concept1: str, concept2: str, fusion_name: str, analyses: Dict) -> float:
        """Calculate overall synthesis quality score"""
        
        # Factor 1: Semantic preservation
        fusion_analysis = self.analyze_semantic_structure(fusion_name)
        domains1 = set(analyses[concept1]['domains'])
        domains2 = set(analyses[concept2]['domains'])
        fusion_domains = set(fusion_analysis['domains'])
        
        preserved_domains = len((domains1 | domains2) & fusion_domains)
        total_domains = len(domains1 | domains2) or 1
        preservation_score = preserved_domains / total_domains
        
        # Factor 2: Novelty (not just concatenation)
        novelty_score = self._calculate_novelty_score(fusion_name)
        
        # Factor 3: Coherence (meaningful combination)
        coherence_score = self._calculate_coherence_score(fusion_name, [concept1, concept2])
        
        # Weighted combination
        synthesis_score = (
            preservation_score * 0.4 +
            novelty_score * 0.3 +
            coherence_score * 0.3
        )
        
        return min(1.0, synthesis_score)
    
    def _calculate_coherence_score(self, fusion_name: str, parents: List[str]) -> float:
        """Calculate how coherent the fusion is with its parents"""
        
        fusion_tokens = set(self._tokenize_concept(fusion_name))
        parent_tokens = set()
        for parent in parents:
            parent_tokens.update(self._tokenize_concept(parent))
        
        # Check token overlap
        overlap = len(fusion_tokens & parent_tokens)
        total_fusion_tokens = len(fusion_tokens) or 1
        
        token_coherence = overlap / total_fusion_tokens
        
        # Check structural coherence (meaningful relationships)
        structure_indicators = ['adaptive', 'meta', 'bridge', 'guided', 'enhanced', 'integrated']
        structure_bonus = 0.2 if any(indicator in fusion_name for indicator in structure_indicators) else 0.0
        
        return min(1.0, token_coherence + structure_bonus)
    
    def _calculate_novelty_score(self, fusion_name: str) -> float:
        """Calculate how novel/creative the fusion is"""
        
        # Check if it's just simple concatenation
        if len(fusion_name.split('-')) == 2 and fusion_name.count('-') == 1:
            concatenation_penalty = 0.3
        else:
            concatenation_penalty = 0.0
        
        # Reward creative terms
        creative_indicators = ['adaptive', 'meta', 'bridge', 'synthesis', 'hybrid', 'emergent', 'coherent']
        creativity_bonus = 0.2 * sum(1 for indicator in creative_indicators if indicator in fusion_name)
        
        # Base novelty (assume moderate)
        base_novelty = 0.5
        
        return min(1.0, base_novelty - concatenation_penalty + creativity_bonus)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize fusion name to canonical form"""
        return name.lower().strip().replace(' ', '-')
    
    def _generate_hash(self, name: str) -> str:
        """Generate unique hash for fusion"""
        import hashlib
        return hashlib.md5(name.encode()).hexdigest()[:16]
    
    async def synthesize_cross_domain_concepts(self, domain_concepts: Dict[str, List[str]]) -> List[Dict]:
        """
        Synthesize concepts that bridge different knowledge domains.
        """
        logger.info(f"🌐 Synthesizing cross-domain concepts from {len(domain_concepts)} domains...")
        
        cross_domain_fusions = []
        
        # Get all domain pairs
        domains = list(domain_concepts.keys())
        
        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                concepts1 = domain_concepts[domain1]
                concepts2 = domain_concepts[domain2]
                
                # Sample concepts from each domain
                sample1 = random.sample(concepts1, min(3, len(concepts1)))
                sample2 = random.sample(concepts2, min(3, len(concepts2)))
                
                # Create cross-domain fusions
                for c1 in sample1:
                    for c2 in sample2:
                        fusion = await self._create_cross_domain_fusion(c1, c2, domain1, domain2)
                        if fusion:
                            cross_domain_fusions.append(fusion)
        
        # Sort by synthesis quality
        cross_domain_fusions.sort(key=lambda x: x.get('synthesis_score', 0), reverse=True)
        
        logger.info(f"✅ Generated {len(cross_domain_fusions)} cross-domain concepts")
        return cross_domain_fusions[:15]  # Return top 15
    
    async def _create_cross_domain_fusion(self, concept1: str, concept2: str, domain1: str, domain2: str) -> Optional[Dict]:
        """Create fusion that bridges two different domains"""
        
        # Create domain-aware fusion name
        domain_bridge_name = f"{domain1}-{domain2}-bridge"
        
        # Extract key characteristics from each concept
        tokens1 = self._tokenize_concept(concept1)
        tokens2 = self._tokenize_concept(concept2)
        
        # Create meaningful bridge concept
        if domain1 == 'cognitive' and domain2 == 'quantum':
            bridge_name = f"quantum-cognitive-{tokens1[0]}-{tokens2[0]}"
        elif domain1 == 'neural' and domain2 == 'computational':
            bridge_name = f"neurocomputational-{tokens1[0]}-{tokens2[0]}"
        elif domain1 == 'dynamic' and domain2 == 'mathematical':
            bridge_name = f"dynamical-{tokens1[0]}-{tokens2[0]}-system"
        else:
            # Generic cross-domain fusion
            bridge_name = f"hybrid-{tokens1[0]}-{tokens2[0]}-interface"
        
        # Calculate synthesis metrics
        synthesis_score = self._calculate_cross_domain_score(concept1, concept2, domain1, domain2)
        
        return {
            'canonical_name': self._normalize_name(bridge_name),
            'fusion_name': bridge_name,
            'parents': [concept1, concept2],
            'source_domains': [domain1, domain2],
            'synthesis_strategy': 'cross_domain_bridge',
            'synthesis_score': synthesis_score,
            'bridge_type': f"{domain1}-{domain2}",
            'concept_hash': self._generate_hash(bridge_name),
            'creation_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_cross_domain_score(self, concept1: str, concept2: str, domain1: str, domain2: str) -> float:
        """Calculate quality score for cross-domain fusion"""
        
        # Base score for successful domain bridging
        base_score = 0.6
        
        # Bonus for complementary domains
        complementary_pairs = [
            ('cognitive', 'computational'),
            ('neural', 'quantum'),
            ('dynamic', 'mathematical'),
            ('neural', 'cognitive')
        ]
        
        if (domain1, domain2) in complementary_pairs or (domain2, domain1) in complementary_pairs:
            complementary_bonus = 0.3
        else:
            complementary_bonus = 0.1
        
        # Concept quality factor
        analysis1 = self.analyze_semantic_structure(concept1)
        analysis2 = self.analyze_semantic_structure(concept2)
        
        avg_complexity = (analysis1['complexity'] + analysis2['complexity']) / 2
        quality_factor = avg_complexity * 0.1
        
        return min(1.0, base_score + complementary_bonus + quality_factor)
    
    async def create_emergent_abstractions(self, concept_clusters: List[List[str]]) -> List[Dict]:
        """
        Create emergent abstract concepts from clusters of related concepts.
        """
        logger.info(f"🚀 Creating emergent abstractions from {len(concept_clusters)} clusters...")
        
        abstractions = []
        
        for i, cluster in enumerate(concept_clusters):
            if len(cluster) < 3:  # Need at least 3 concepts for meaningful abstraction
                continue
            
            abstraction = await self._create_cluster_abstraction(cluster, i)
            if abstraction:
                abstractions.append(abstraction)
        
        logger.info(f"✅ Generated {len(abstractions)} emergent abstractions")
        return abstractions
    
    async def _create_cluster_abstraction(self, cluster: List[str], cluster_id: int) -> Optional[Dict]:
        """Create an abstract concept from a cluster of related concepts"""
        
        # Analyze all concepts in cluster
        cluster_analyses = [self.analyze_semantic_structure(concept) for concept in cluster]
        
        # Find common themes
        all_tokens = []
        all_domains = []
        
        for analysis in cluster_analyses:
            all_tokens.extend(analysis['tokens'])
            all_domains.extend(analysis['domains'])
        
        # Find most frequent meaningful tokens
        token_freq = Counter(all_tokens)
        common_tokens = [token for token, freq in token_freq.most_common(3) if freq > 1]
        
        # Find dominant domains
        domain_freq = Counter(all_domains)
        dominant_domains = [domain for domain, freq in domain_freq.most_common(2)]
        
        # Create abstraction name
        if common_tokens and dominant_domains:
            if len(dominant_domains) == 1:
                abstraction_name = f"{dominant_domains[0]}-{common_tokens[0]}-framework"
            else:
                abstraction_name = f"{dominant_domains[0]}-{dominant_domains[1]}-{common_tokens[0]}-system"
        else:
            abstraction_name = f"emergent-cluster-{cluster_id}-abstraction"
        
        # Calculate abstraction quality
        abstraction_score = self._calculate_abstraction_quality(cluster, abstraction_name)
        
        return {
            'canonical_name': self._normalize_name(abstraction_name),
            'abstraction_name': abstraction_name,
            'source_cluster': cluster,
            'cluster_size': len(cluster),
            'common_themes': common_tokens,
            'dominant_domains': dominant_domains,
            'synthesis_strategy': 'emergent_abstraction',
            'abstraction_score': abstraction_score,
            'concept_hash': self._generate_hash(abstraction_name),
            'creation_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_abstraction_quality(self, cluster: List[str], abstraction_name: str) -> float:
        """Calculate quality of abstraction"""
        
        # Factor 1: Cluster coherence
        cluster_coherence = self._calculate_cluster_coherence(cluster)
        
        # Factor 2: Abstraction appropriateness
        abstraction_tokens = self._tokenize_concept(abstraction_name)
        abstract_indicators = ['framework', 'system', 'model', 'theory', 'principle']
        abstraction_quality = 0.3 if any(indicator in abstraction_tokens for indicator in abstract_indicators) else 0.1
        
        # Factor 3: Coverage (how well abstraction represents cluster)
        coverage_score = self._calculate_abstraction_coverage(cluster, abstraction_name)
        
        return (cluster_coherence * 0.4 + abstraction_quality + coverage_score * 0.3)
    
    def _calculate_cluster_coherence(self, cluster: List[str]) -> float:
        """Calculate how coherent a cluster of concepts is"""
        if len(cluster) < 2:
            return 1.0
        
        # Calculate pairwise semantic similarities
        similarities = []
        for i, c1 in enumerate(cluster):
            for c2 in cluster[i+1:]:
                similarity = self._semantic_similarity(c1, c2)
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _semantic_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate semantic similarity between two concepts"""
        tokens1 = set(self._tokenize_concept(concept1))
        tokens2 = set(self._tokenize_concept(concept2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_abstraction_coverage(self, cluster: List[str], abstraction_name: str) -> float:
        """Calculate how well abstraction covers the source cluster"""
        abstraction_tokens = set(self._tokenize_concept(abstraction_name))
        
        covered_concepts = 0
        for concept in cluster:
            concept_tokens = set(self._tokenize_concept(concept))
            if abstraction_tokens & concept_tokens:  # Any overlap
                covered_concepts += 1
        
        return covered_concepts / len(cluster) if cluster else 0.0
    
    def get_synthesis_stats(self) -> Dict:
        """Get comprehensive synthesis statistics"""
        return {
            'synthesis_cycles': len(self.synthesis_history),
            'semantic_cache_size': len(self.semantic_cache),
            'known_domains': list(self.domain_keywords.keys()),
            'synthesis_parameters': {
                'creativity_factor': self.creativity_factor,
                'semantic_threshold': self.semantic_threshold,
                'coherence_weight': self.coherence_weight,
                'novelty_weight': self.novelty_weight
            },
            'last_synthesis': self.synthesis_history[-1] if self.synthesis_history else None
        }

if __name__ == "__main__":
    # Test Concept Synthesizer
    import asyncio
    
    async def test_concept_synthesizer():
        synthesizer = ConceptSynthesizer()
        
        # Test semantic fusion
        test_concepts = ['neural network', 'cognitive model', 'phase synchrony', 'quantum field']
        fusions = await synthesizer.synthesize_semantic_fusion(test_concepts)
        print(f"🧬 Semantic fusions: {len(fusions)}")
        for fusion in fusions[:3]:
            print(f"  - {fusion['fusion_name']} (score: {fusion['synthesis_score']:.3f})")
        
        # Test cross-domain synthesis
        domain_concepts = {
            'cognitive': ['memory model', 'attention mechanism'],
            'quantum': ['wave function', 'coherent state'],
            'neural': ['synaptic plasticity', 'neural oscillation']
        }
        
        cross_domain = await synthesizer.synthesize_cross_domain_concepts(domain_concepts)
        print(f"🌐 Cross-domain concepts: {len(cross_domain)}")
        for concept in cross_domain[:3]:
            print(f"  - {concept['fusion_name']} ({concept['source_domains']})")
        
        # Test emergent abstractions
        clusters = [
            ['neural network', 'synaptic plasticity', 'neural oscillation'],
            ['quantum field', 'wave function', 'coherent state']
        ]
        
        abstractions = await synthesizer.create_emergent_abstractions(clusters)
        print(f"🚀 Emergent abstractions: {len(abstractions)}")
        for abstraction in abstractions:
            print(f"  - {abstraction['abstraction_name']} (domains: {abstraction['dominant_domains']})")
        
        # Get stats
        stats = synthesizer.get_synthesis_stats()
        print(f"📊 Synthesis stats: {stats}")
    
    asyncio.run(test_concept_synthesizer())
