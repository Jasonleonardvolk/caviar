#!/usr/bin/env python3
"""
ArXiv Recursive Tier 1 Miner 🔄
Follows citation trails to find ALL consciousness/AGI/quantum papers!

Strategy:
1. Find Tier 1 papers (consciousness, AGI, quantum)
2. Extract their citations/references  
3. Search for those papers on arXiv
4. Repeat recursively until we have EVERYTHING
5. Build the ultimate consciousness knowledge foundation!
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

# TIER 1 ULTRA-PRIORITY KEYWORDS (The Holy Grail)
TIER_1_KEYWORDS = {
    "consciousness": [
        "consciousness", "conscious", "awareness", "phenomenal consciousness",
        "integrated information theory", "IIT", "global workspace theory", "GWT",
        "attention schema theory", "AST", "higher order thought", "HOT",
        "qualia", "hard problem of consciousness", "explanatory gap",
        "neural correlates of consciousness", "NCC", "consciousness meter"
    ],
    
    "agi": [
        "artificial general intelligence", "AGI", "general intelligence",
        "human-level AI", "superintelligence", "artificial consciousness",
        "machine consciousness", "cognitive architecture", "artificial minds",
        "AI alignment", "friendly AI", "AI safety", "intelligence explosion"
    ],
    
    "quantum_consciousness": [
        "quantum consciousness", "quantum cognition", "quantum mind",
        "orchestrated objective reduction", "ORCH-OR", "quantum brain",
        "quantum neural networks", "quantum information processing",
        "quantum entanglement consciousness", "microtubules consciousness"
    ],
    
    "emergence": [
        "emergence", "emergent consciousness", "emergent intelligence", 
        "complex adaptive systems", "self-organization", "autopoiesis",
        "collective intelligence", "swarm consciousness", "distributed cognition"
    ]
}

# Citation patterns to extract from papers
CITATION_PATTERNS = [
    r'arXiv:(\d{4}\.\d{4,5})',  # arXiv ID format
    r'(\d{4}\.\d{4,5})',        # Just the number part
    r'arxiv\.org/abs/(\d{4}\.\d{4,5})',  # Full URL format
]

class RecursiveTier1Miner:
    def __init__(self):
        self.found_papers = set()  # Track all discovered papers
        self.tier1_papers = set()  # Only tier 1 papers
        self.citation_network = {}  # Track citation relationships
        self.recursion_depth = 0
        self.max_depth = 5  # Prevent infinite recursion
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('recursive_tier1_mining.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_tier1_score(self, paper) -> float:
        """Calculate how 'Tier 1' a paper is"""
        title_lower = paper.title.lower()
        abstract_lower = paper.summary.lower()
        
        score = 0.0
        category_matches = {}
        
        for category, keywords in TIER_1_KEYWORDS.items():
            category_score = 0.0
            for keyword in keywords:
                if keyword in title_lower:
                    category_score += 3.0  # Title matches worth more
                elif keyword in abstract_lower:
                    category_score += 1.0
            
            if category_score > 0:
                category_matches[category] = category_score
                score += category_score
        
        # Bonus for multiple category matches (interdisciplinary)
        if len(category_matches) > 1:
            score += 2.0 * len(category_matches)
        
        return score, category_matches
    
    def extract_citations_from_paper(self, paper) -> Set[str]:
        """Extract arXiv citations from a paper's content"""
        citations = set()
        
        # Search in title and abstract for arXiv IDs
        full_text = f"{paper.title} {paper.summary}"
        
        for pattern in CITATION_PATTERNS:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                # Clean up the arXiv ID
                arxiv_id = match.strip()
                if len(arxiv_id) >= 9:  # Valid arXiv ID length
                    citations.add(arxiv_id)
        
        return citations
    
    def search_arxiv_by_ids(self, arxiv_ids: Set[str]) -> List:
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
    
    def recursive_tier1_search(self, seed_papers: List, depth: int = 0) -> List:
        """Recursively search for all Tier 1 papers"""
        if depth >= self.max_depth:
            self.logger.info(f"🛑 Max recursion depth {self.max_depth} reached")
            return []
        
        self.logger.info(f"🔄 Recursion level {depth}: Processing {len(seed_papers)} papers")
        
        new_tier1_papers = []
        all_citations = set()
        
        # Process each paper in this batch
        for paper in seed_papers:
            # Check if it's Tier 1 quality
            tier1_score, categories = self.calculate_tier1_score(paper)
            
            if tier1_score >= 3.0:  # Threshold for Tier 1
                new_tier1_papers.append({
                    'paper': paper,
                    'score': tier1_score,
                    'categories': categories,
                    'depth': depth
                })
                
                self.tier1_papers.add(paper.entry_id)
                self.logger.info(f"⭐ Tier 1 found [{tier1_score:.1f}★]: {paper.title[:80]}...")
                
                # Extract citations from this Tier 1 paper
                citations = self.extract_citations_from_paper(paper)
                all_citations.update(citations)
                
                # Track citation network
                self.citation_network[paper.entry_id] = {
                    'title': paper.title,
                    'score': tier1_score,
                    'citations': list(citations),
                    'depth': depth
                }
        
        # If we found new citations, search for them recursively
        if all_citations and depth < self.max_depth:
            self.logger.info(f"🔍 Found {len(all_citations)} citations to explore at depth {depth+1}")
            
            # Filter out already found papers
            new_citations = all_citations - {p.split('/')[-1] for p in self.found_papers}
            
            if new_citations:
                # Search for cited papers
                cited_papers = self.search_arxiv_by_ids(new_citations)
                
                if cited_papers:
                    # Recursively search the cited papers
                    deeper_papers = self.recursive_tier1_search(cited_papers, depth + 1)
                    new_tier1_papers.extend(deeper_papers)
        
        return new_tier1_papers
    
    def initial_tier1_seed_search(self) -> List:
        """Find initial seed of Tier 1 papers"""
        self.logger.info("🌱 Starting initial seed search for Tier 1 papers...")
        
        seed_papers = []
        
        # Search for each category of Tier 1 keywords
        for category, keywords in TIER_1_KEYWORDS.items():
            self.logger.info(f"🔍 Searching for {category} papers...")
            
            # Create search queries for this category
            for keyword in keywords[:3]:  # Top 3 keywords per category
                try:
                    search = arxiv.Search(
                        query=f'all:"{keyword}"',
                        max_results=50,  # More conservative for seed
                        sort_by=arxiv.SortCriterion.Relevance,
                        sort_order=arxiv.SortOrder.Descending
                    )
                    
                    for paper in search.results():
                        if paper.entry_id not in self.found_papers:
                            seed_papers.append(paper)
                            self.found_papers.add(paper.entry_id)
                    
                    time.sleep(2)  # Rate limiting
                    
                except Exception as e:
                    self.logger.error(f"Error searching for '{keyword}': {e}")
        
        self.logger.info(f"🌱 Initial seed: {len(seed_papers)} papers found")
        return seed_papers
    
    def save_results(self, all_tier1_papers: List):
        """Save the complete Tier 1 paper collection"""
        
        # Prepare results
        results = {
            'total_tier1_papers': len(all_tier1_papers),
            'recursion_depths': {},
            'category_breakdown': {},
            'citation_network': self.citation_network,
            'papers': []
        }
        
        # Analyze results
        for paper_info in all_tier1_papers:
            depth = paper_info['depth']
            score = paper_info['score']
            categories = paper_info['categories']
            
            # Track depth distribution
            if depth not in results['recursion_depths']:
                results['recursion_depths'][depth] = 0
            results['recursion_depths'][depth] += 1
            
            # Track category distribution
            for category in categories.keys():
                if category not in results['category_breakdown']:
                    results['category_breakdown'][category] = 0
                results['category_breakdown'][category] += 1
            
            # Store paper info
            results['papers'].append({
                'title': paper_info['paper'].title,
                'arxiv_id': paper_info['paper'].entry_id,
                'score': score,
                'categories': categories,
                'depth': depth,
                'published': paper_info['paper'].published.isoformat() if paper_info['paper'].published else None
            })
        
        # Save to file
        with open('recursive_tier1_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def run_complete_recursive_search(self):
        """Run the complete recursive Tier 1 mining operation"""
        self.logger.info("🚀 STARTING RECURSIVE TIER 1 MINING")
        self.logger.info("🎯 Goal: Find ALL consciousness/AGI/quantum papers on arXiv")
        self.logger.info("🔄 Method: Citation-driven recursive search")
        print("="*60)
        
        start_time = time.time()
        
        # Step 1: Find initial seed papers
        seed_papers = self.initial_tier1_seed_search()
        
        # Step 2: Recursive search following citations
        all_tier1_papers = self.recursive_tier1_search(seed_papers)
        
        # Step 3: Save and analyze results
        results = self.save_results(all_tier1_papers)
        
        elapsed = time.time() - start_time
        
        # Print final results
        self.logger.info("🎉 " + "="*50)
        self.logger.info("🎉 RECURSIVE TIER 1 MINING COMPLETE!")
        self.logger.info("🎉 " + "="*50)
        self.logger.info(f"📚 Total Tier 1 papers found: {len(all_tier1_papers)}")
        self.logger.info(f"⏱️ Total mining time: {elapsed/60:.1f} minutes")
        self.logger.info(f"🔄 Recursion depths used: {list(results['recursion_depths'].keys())}")
        self.logger.info(f"📊 Category breakdown:")
        for category, count in results['category_breakdown'].items():
            self.logger.info(f"   {category}: {count} papers")
        self.logger.info(f"🕸️ Citation network: {len(self.citation_network)} connected papers")
        self.logger.info("🎉 " + "="*50)
        
        return all_tier1_papers

def main():
    """Start the recursive mining beast!"""
    print("🔄 RECURSIVE TIER 1 MINER - The Ultimate Knowledge Hunter!")
    print("🎯 Mission: Find EVERY consciousness/AGI/quantum paper on arXiv")
    print("🕸️ Method: Follow citation trails recursively")
    print("🍺 Perfect for a long beer session - this will take HOURS!")
    print("="*60)
    
    miner = RecursiveTier1Miner()
    tier1_papers = miner.run_complete_recursive_search()
    
    print(f"🏆 Mission accomplished! {len(tier1_papers)} Tier 1 papers discovered!")
    print("🧠 Your consciousness knowledge foundation is now LEGENDARY!")

if __name__ == "__main__":
    main()
