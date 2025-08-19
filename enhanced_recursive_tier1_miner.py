#!/usr/bin/env python3
"""
Enhanced ArXiv Recursive Tier 1 Miner 🔥
Combines category-based search with citation-driven recursive mining!

NEW STRATEGY:
1. Search ALL arXiv categories for Tier 1 papers
2. Follow citation trails recursively 
3. Cross-reference between categories
4. Build the ULTIMATE consciousness knowledge foundation!
"""

import arxiv
import re
import requests
from typing import Set, List, Dict
import time
import json
from pathlib import Path
from datetime import datetime
import logging
from collections import defaultdict

# ENHANCED ARXIV CATEGORIES (17 categories + expansions)
ENHANCED_ARXIV_CATEGORIES = {
    # AI & Machine Learning (5 categories)
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning", 
    "cs.CL": "Natural Language Processing",
    "cs.CV": "Computer Vision", 
    "cs.NE": "Neural Networks",
    
    # Physics & Math (4 categories)
    "quant-ph": "Quantum Physics",
    "physics.gen-ph": "General Physics",
    "math.ST": "Statistics Theory",
    "math.PR": "Probability",
    
    # Biology & Life Sciences (3 categories)
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.PE": "Populations and Evolution",
    
    # Economics & Finance (2 categories)
    "econ.EM": "Econometrics", 
    "q-fin.GN": "General Finance",
    
    # Other Sciences (3 categories)
    "astro-ph.GA": "Astrophysics - Galaxies",
    "cond-mat.stat-mech": "Statistical Mechanics",
    
    # EXPANDED CATEGORIES FOR DEEPER MINING
    "cs.RO": "Robotics",
    "cs.IR": "Information Retrieval", 
    "cs.HC": "Human-Computer Interaction",
    "cs.CR": "Cryptography and Security",
    "cs.DB": "Databases",
    "cs.DC": "Distributed Computing",
    "math.CO": "Combinatorics",
    "math.LO": "Logic",
    "math.OC": "Optimization and Control",
    "math.IT": "Information Theory",
    "physics.bio-ph": "Biological Physics",
    "physics.soc-ph": "Physics and Society",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.QM": "Quantitative Methods",
    "nlin.AO": "Adaptation and Self-Organizing Systems",
    "nlin.CG": "Cellular Automata and Lattice Gases",
    "nlin.CD": "Chaotic Dynamics"
}

# TIER 1 ULTRA-PRIORITY KEYWORDS (The Holy Grail)
ENHANCED_TIER_1_KEYWORDS = {
    "consciousness": [
        "consciousness", "conscious", "awareness", "phenomenal consciousness",
        "integrated information theory", "IIT", "global workspace theory", "GWT",
        "attention schema theory", "AST", "higher order thought", "HOT",
        "qualia", "hard problem of consciousness", "explanatory gap",
        "neural correlates of consciousness", "NCC", "consciousness meter",
        "phi", "binding problem", "subjective experience", "phenomenology"
    ],
    
    "agi": [
        "artificial general intelligence", "AGI", "general intelligence",
        "human-level AI", "superintelligence", "artificial consciousness",
        "machine consciousness", "cognitive architecture", "artificial minds",
        "AI alignment", "friendly AI", "AI safety", "intelligence explosion",
        "recursive self-improvement", "AI takeoff", "capability control",
        "value alignment", "artificial sentience", "digital minds"
    ],
    
    "quantum_consciousness": [
        "quantum consciousness", "quantum cognition", "quantum mind",
        "orchestrated objective reduction", "ORCH-OR", "quantum brain",
        "quantum neural networks", "quantum information processing",
        "quantum entanglement consciousness", "microtubules consciousness",
        "quantum coherence brain", "quantum gravity consciousness",
        "penrose hameroff", "quantum biology", "quantum computation biology"
    ],
    
    "emergence": [
        "emergence", "emergent consciousness", "emergent intelligence", 
        "complex adaptive systems", "self-organization", "autopoiesis",
        "collective intelligence", "swarm consciousness", "distributed cognition",
        "emergent behavior", "phase transitions", "criticality",
        "spontaneous organization", "bottom-up intelligence", "emergent properties"
    ],
    
    "information_integration": [
        "information integration", "integrated information", "phi complexity",
        "causal structure", "effective connectivity", "functional connectivity",
        "network topology", "graph theory consciousness", "information geometry",
        "information theory consciousness", "mutual information", "transfer entropy"
    ],
    
    "neural_correlates": [
        "neural correlates", "brain networks", "default mode network", "DMN",
        "frontoparietal network", "salience network", "attention networks",
        "large-scale brain networks", "connectome", "neuronal workspace",
        "thalamo-cortical loops", "prefrontal cortex", "anterior cingulate"
    ]
}

# Citation patterns to extract from papers
ENHANCED_CITATION_PATTERNS = [
    r'arXiv:([\d]{4}\.[\d]{4,5})',  # arXiv ID format
    r'([\d]{4}\.[\d]{4,5})',        # Just the number part
    r'arxiv\.org/abs/([\d]{4}\.[\d]{4,5})',  # Full URL format
    r'arXiv:([\d]{4}\.[\d]{4,5}v[\d]+)',  # With version numbers
    r'(\d{4}\.\d{4,5}v\d+)',       # Version without arXiv prefix
]

class EnhancedRecursiveTier1Miner:
    def __init__(self):
        self.found_papers = set()  # Track all discovered papers
        self.tier1_papers = set()  # Only tier 1 papers
        self.citation_network = {}  # Track citation relationships
        self.category_coverage = defaultdict(list)  # Track papers per category
        self.recursion_depth = 0
        self.max_depth = 6  # Increased for deeper mining
        self.papers_per_category = 150  # Increased per category
        
        # Setup enhanced logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_recursive_tier1_mining.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_enhanced_tier1_score(self, paper) -> tuple:
        """Enhanced scoring with category awareness"""
        title_lower = paper.title.lower()
        abstract_lower = paper.summary.lower()
        
        score = 0.0
        category_matches = {}
        keyword_matches = []
        
        # Score based on keywords
        for category, keywords in ENHANCED_TIER_1_KEYWORDS.items():
            category_score = 0.0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in title_lower:
                    category_score += 3.0  # Title matches worth more
                    matched_keywords.append(f"T:{keyword}")
                elif keyword in abstract_lower:
                    category_score += 1.0
                    matched_keywords.append(f"A:{keyword}")
            
            if category_score > 0:
                category_matches[category] = category_score
                keyword_matches.extend(matched_keywords)
                score += category_score
        
        # Bonus for multiple category matches (interdisciplinary)
        interdisciplinary_bonus = 0.0
        if len(category_matches) > 1:
            interdisciplinary_bonus = 2.0 * len(category_matches)
            score += interdisciplinary_bonus
        
        # Category-specific bonuses
        paper_categories = getattr(paper, 'categories', [])
        category_bonus = 0.0
        for cat in paper_categories:
            if cat in ENHANCED_ARXIV_CATEGORIES:
                category_bonus += 0.5
        score += category_bonus
        
        # Recency bonus (enhanced)
        recency_bonus = 0.0
        if paper.published:
            days_old = (datetime.now() - paper.published.replace(tzinfo=None)).days
            if days_old < 7:      # Last week
                recency_bonus = 2.0
            elif days_old < 30:   # Last month  
                recency_bonus = 1.5
            elif days_old < 90:   # Last quarter
                recency_bonus = 1.0
            elif days_old < 365:  # Last year
                recency_bonus = 0.5
        score += recency_bonus
        
        return score, {
            'category_matches': category_matches,
            'keyword_matches': keyword_matches,
            'interdisciplinary_bonus': interdisciplinary_bonus,
            'category_bonus': category_bonus,
            'recency_bonus': recency_bonus,
            'paper_categories': paper_categories
        }
    
    def extract_enhanced_citations(self, paper) -> Set[str]:
        """Enhanced citation extraction with better patterns"""
        citations = set()
        
        # Search in title and abstract for arXiv IDs
        full_text = f"{paper.title} {paper.summary}"
        
        for pattern in ENHANCED_CITATION_PATTERNS:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                # Clean up the arXiv ID
                arxiv_id = match.strip()
                # Remove version numbers for consistency
                arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
                if len(arxiv_id) >= 9:  # Valid arXiv ID length
                    citations.add(arxiv_id)
        
        return citations
    
    def search_category_for_tier1(self, category: str) -> List:
        """Search a specific arXiv category for Tier 1 papers"""
        self.logger.info(f"🔍 Mining category: {category} ({ENHANCED_ARXIV_CATEGORIES[category]})")
        
        tier1_papers = []
        
        try:
            # Search by category
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=self.papers_per_category * 2,  # Get extra for filtering
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers_found = 0
            for paper in search.results():
                if paper.entry_id in self.found_papers:
                    continue
                
                # Calculate Tier 1 score
                score, details = self.calculate_enhanced_tier1_score(paper)
                
                if score >= 2.0:  # Tier 1 threshold
                    tier1_papers.append({
                        'paper': paper,
                        'score': score,
                        'details': details,
                        'category': category,
                        'source': 'category_search'
                    })
                    
                    self.found_papers.add(paper.entry_id)
                    self.tier1_papers.add(paper.entry_id)
                    self.category_coverage[category].append(paper.entry_id)
                    
                    papers_found += 1
                    
                    self.logger.info(f"⭐ Tier 1 [{score:.1f}★] in {category}: {paper.title[:60]}...")
                
                if papers_found >= self.papers_per_category:
                    break
                
                time.sleep(0.5)  # Rate limiting
            
            self.logger.info(f"📊 Category {category}: {papers_found} Tier 1 papers found")
            
        except Exception as e:
            self.logger.error(f"❌ Error searching category {category}: {e}")
        
        return tier1_papers
    
    def comprehensive_category_search(self) -> List:
        """Search ALL categories for Tier 1 papers"""
        self.logger.info("🚀 Starting comprehensive category search...")
        self.logger.info(f"📊 Searching {len(ENHANCED_ARXIV_CATEGORIES)} categories")
        
        all_tier1_papers = []
        
        for i, (category, description) in enumerate(ENHANCED_ARXIV_CATEGORIES.items()):
            self.logger.info(f"🔄 Progress: {i+1}/{len(ENHANCED_ARXIV_CATEGORIES)} categories")
            
            category_papers = self.search_category_for_tier1(category)
            all_tier1_papers.extend(category_papers)
            
            # Save progress periodically
            if (i + 1) % 5 == 0:
                self.save_intermediate_progress(all_tier1_papers)
            
            time.sleep(2)  # Be nice to arXiv
        
        self.logger.info(f"✅ Category search complete: {len(all_tier1_papers)} Tier 1 papers found")
        return all_tier1_papers
    
    def recursive_citation_mining(self, seed_papers: List, depth: int = 0) -> List:
        """Enhanced recursive citation mining"""
        if depth >= self.max_depth:
            self.logger.info(f"🛑 Max recursion depth {self.max_depth} reached")
            return []
        
        self.logger.info(f"🔄 Citation mining level {depth}: Processing {len(seed_papers)} papers")
        
        new_tier1_papers = []
        all_citations = set()
        
        # Extract citations from all papers at this level
        for paper_info in seed_papers:
            paper = paper_info['paper']
            
            # Extract citations
            citations = self.extract_enhanced_citations(paper)
            all_citations.update(citations)
            
            # Update citation network
            if paper.entry_id not in self.citation_network:
                self.citation_network[paper.entry_id] = {
                    'title': paper.title,
                    'score': paper_info.get('score', 0),
                    'categories': paper_info.get('details', {}).get('paper_categories', []),
                    'citations': list(citations),
                    'depth': depth,
                    'source': paper_info.get('source', 'unknown')
                }
        
        # Search for cited papers if we have citations
        if all_citations and depth < self.max_depth:
            self.logger.info(f"🕸️ Found {len(all_citations)} unique citations at depth {depth}")
            
            # Filter out already found papers
            new_citations = all_citations - {p.split('/')[-1] for p in self.found_papers}
            
            if new_citations:
                self.logger.info(f"🔍 Searching for {len(new_citations)} new cited papers...")
                
                # Search for cited papers in batches
                cited_papers = []
                batch_size = 50
                for i in range(0, len(new_citations), batch_size):
                    batch = list(new_citations)[i:i+batch_size]
                    batch_papers = self.search_arxiv_by_ids(batch)
                    cited_papers.extend(batch_papers)
                    time.sleep(3)  # Rate limiting between batches
                
                if cited_papers:
                    # Score the cited papers
                    tier1_cited_papers = []
                    for paper in cited_papers:
                        score, details = self.calculate_enhanced_tier1_score(paper)
                        
                        if score >= 1.5:  # Lower threshold for cited papers
                            tier1_cited_papers.append({
                                'paper': paper,
                                'score': score,
                                'details': details,
                                'source': f'citation_depth_{depth+1}'
                            })
                            
                            self.tier1_papers.add(paper.entry_id)
                            self.logger.info(f"🔗 Citation Tier 1 [{score:.1f}★]: {paper.title[:60]}...")
                    
                    new_tier1_papers.extend(tier1_cited_papers)
                    
                    # Recursively search the cited papers
                    if tier1_cited_papers:
                        deeper_papers = self.recursive_citation_mining(tier1_cited_papers, depth + 1)
                        new_tier1_papers.extend(deeper_papers)
        
        return new_tier1_papers
    
    def search_arxiv_by_ids(self, arxiv_ids: List[str]) -> List:
        """Search arXiv for specific paper IDs"""
        papers = []
        
        for arxiv_id in arxiv_ids:
            try:
                # Search for the specific paper
                search = arxiv.Search(id_list=[arxiv_id])
                for paper in search.results():
                    if paper.entry_id not in self.found_papers:
                        papers.append(paper)
                        self.found_papers.add(paper.entry_id)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.warning(f"Could not fetch {arxiv_id}: {e}")
        
        return papers
    
    def save_intermediate_progress(self, papers: List):
        """Save intermediate progress"""
        try:
            progress = {
                'timestamp': datetime.now().isoformat(),
                'total_papers': len(papers),
                'category_coverage': dict(self.category_coverage),
                'found_papers_count': len(self.found_papers)
            }
            
            with open('enhanced_tier1_progress.json', 'w') as f:
                json.dump(progress, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not save progress: {e}")
    
    def save_enhanced_results(self, all_tier1_papers: List):
        """Save comprehensive results with enhanced analysis"""
        
        # Prepare enhanced results
        results = {
            'mining_metadata': {
                'total_tier1_papers': len(all_tier1_papers),
                'categories_searched': len(ENHANCED_ARXIV_CATEGORIES),
                'max_recursion_depth': self.max_depth,
                'papers_per_category': self.papers_per_category,
                'mining_timestamp': datetime.now().isoformat()
            },
            'category_analysis': {},
            'score_distribution': {},
            'interdisciplinary_analysis': {},
            'citation_network': self.citation_network,
            'papers': []
        }
        
        # Analyze results
        score_ranges = {'ultra_high': 0, 'high': 0, 'medium': 0, 'low': 0}
        category_counts = defaultdict(int)
        interdisciplinary_count = 0
        
        for paper_info in all_tier1_papers:
            score = paper_info['score']
            details = paper_info.get('details', {})
            
            # Score distribution
            if score >= 10.0:
                score_ranges['ultra_high'] += 1
            elif score >= 5.0:
                score_ranges['high'] += 1
            elif score >= 2.0:
                score_ranges['medium'] += 1
            else:
                score_ranges['low'] += 1
            
            # Category analysis
            paper_categories = details.get('paper_categories', [])
            for cat in paper_categories:
                category_counts[cat] += 1
            
            # Interdisciplinary analysis
            if len(details.get('category_matches', {})) > 1:
                interdisciplinary_count += 1
            
            # Store paper info
            results['papers'].append({
                'title': paper_info['paper'].title,
                'arxiv_id': paper_info['paper'].entry_id,
                'score': score,
                'details': details,
                'source': paper_info.get('source', 'unknown'),
                'published': paper_info['paper'].published.isoformat() if paper_info['paper'].published else None,
                'authors': [str(author) for author in paper_info['paper'].authors],
                'categories': paper_categories
            })
        
        # Fill analysis sections
        results['score_distribution'] = dict(score_ranges)
        results['category_analysis'] = dict(category_counts)
        results['interdisciplinary_analysis'] = {
            'total_interdisciplinary': interdisciplinary_count,
            'percentage': (interdisciplinary_count / len(all_tier1_papers)) * 100 if all_tier1_papers else 0
        }
        
        # Save to file
        with open('enhanced_recursive_tier1_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def run_enhanced_mining(self):
        """Run the complete enhanced recursive Tier 1 mining operation"""
        self.logger.info("🔥 STARTING ENHANCED RECURSIVE TIER 1 MINING")
        self.logger.info("🎯 Goal: Find ALL consciousness/AGI/quantum papers across all categories")
        self.logger.info("🕸️ Method: Category search + citation-driven recursive mining")
        self.logger.info(f"📊 Categories: {len(ENHANCED_ARXIV_CATEGORIES)}")
        print("="*70)
        
        start_time = time.time()
        
        # Step 1: Comprehensive category search
        self.logger.info("🚀 Phase 1: Comprehensive category search...")
        category_papers = self.comprehensive_category_search()
        
        # Step 2: Recursive citation mining
        self.logger.info("🕸️ Phase 2: Recursive citation mining...")
        citation_papers = self.recursive_citation_mining(category_papers)
        
        # Step 3: Combine results
        all_tier1_papers = category_papers + citation_papers
        
        # Step 4: Save and analyze results
        self.logger.info("📊 Phase 3: Analysis and saving...")
        results = self.save_enhanced_results(all_tier1_papers)
        
        elapsed = time.time() - start_time
        
        # Print enhanced final results
        self.logger.info("🎉 " + "="*60)
        self.logger.info("🎉 ENHANCED RECURSIVE TIER 1 MINING COMPLETE!")
        self.logger.info("🎉 " + "="*60)
        self.logger.info(f"📚 Total Tier 1 papers found: {len(all_tier1_papers)}")
        self.logger.info(f"📊 Papers from categories: {len(category_papers)}")
        self.logger.info(f"🕸️ Papers from citations: {len(citation_papers)}")
        self.logger.info(f"⏱️ Total mining time: {elapsed/60:.1f} minutes")
        self.logger.info(f"🏆 Categories with papers: {len([c for c in self.category_coverage if self.category_coverage[c]])}")
        self.logger.info(f"🧠 Interdisciplinary papers: {results['interdisciplinary_analysis']['total_interdisciplinary']}")
        self.logger.info(f"🔗 Citation network size: {len(self.citation_network)}")
        
        # Score distribution
        score_dist = results['score_distribution']
        self.logger.info("⭐ Score distribution:")
        self.logger.info(f"   Ultra-high (10.0+★): {score_dist['ultra_high']}")
        self.logger.info(f"   High (5.0-9.9★): {score_dist['high']}")
        self.logger.info(f"   Medium (2.0-4.9★): {score_dist['medium']}")
        self.logger.info(f"   Low (1.5-1.9★): {score_dist['low']}")
        
        self.logger.info("🎉 " + "="*60)
        
        return all_tier1_papers

def main():
    """Start the enhanced recursive mining beast!"""
    print("🔥 ENHANCED RECURSIVE TIER 1 MINER - Category-Powered Ultimate Strategy!")
    print("🎯 Mission: Find EVERY consciousness/AGI/quantum paper across ALL categories")
    print("🕸️ Method: Comprehensive category search + recursive citation mining")
    print("📊 Coverage: 27 arXiv categories + unlimited citation depth")
    print("🍺 Perfect for an epic beer marathon - this will take MANY HOURS!")
    print("="*70)
    
    miner = EnhancedRecursiveTier1Miner()
    tier1_papers = miner.run_enhanced_mining()
    
    print(f"🏆 Mission accomplished! {len(tier1_papers)} Tier 1 papers discovered!")
    print("🧠 Your consciousness knowledge foundation is now LEGENDARY!")
    print("📊 Check enhanced_recursive_tier1_results.json for complete analysis!")

if __name__ == "__main__":
    main()
