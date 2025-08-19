#!/usr/bin/env python3
"""
Enhanced Tier 1 Paper Downloader 📥
Downloads all papers discovered by the recursive miner!
"""

import json
import os
import arxiv
import time
import logging
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import re
from typing import List, Dict

# Configuration
RESULTS_FILE = "enhanced_recursive_tier1_results.json"
DOWNLOAD_DIR = r"{PROJECT_ROOT}\data\tier1_consciousness_papers"
MAX_CONCURRENT = 3
RATE_LIMIT_DELAY = 2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tier1_paper_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Tier1PaperDownloader:
    def __init__(self):
        self.download_dir = Path(DOWNLOAD_DIR)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_count = 0
        self.failed_count = 0
        
    def load_discovered_papers(self) -> List[Dict]:
        """Load papers from the recursive mining results"""
        try:
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
                papers = data.get('papers', [])
                logger.info(f"📚 Loaded {len(papers)} discovered Tier 1 papers")
                return papers
        except FileNotFoundError:
            logger.error(f"❌ Results file not found: {RESULTS_FILE}")
            logger.info("💡 Run the enhanced recursive miner first!")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading results: {e}")
            return []
    
    def create_safe_filename(self, title: str, arxiv_id: str) -> str:
        """Create a safe filename from paper title"""
        # Clean title for filename
        safe_title = re.sub(r'[^\w\s-]', '', title)[:100]
        safe_title = re.sub(r'\s+', '_', safe_title)
        
        # Extract just the ID part
        id_part = arxiv_id.split('/')[-1].split('v')[0]
        
        return f"{safe_title}_{id_part}.pdf"
    
    def organize_by_category(self, papers: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize papers by their source and score"""
        organized = {
            'ultra_high_priority': [],  # Score 10.0+
            'high_priority': [],        # Score 5.0-9.9
            'medium_priority': [],      # Score 2.0-4.9
            'citation_discoveries': []  # From citation mining
        }
        
        for paper in papers:
            score = paper.get('score', 0)
            source = paper.get('source', '')
            
            if score >= 10.0:
                organized['ultra_high_priority'].append(paper)
            elif score >= 5.0:
                organized['high_priority'].append(paper)
            elif score >= 2.0:
                organized['medium_priority'].append(paper)
            elif 'citation' in source:
                organized['citation_discoveries'].append(paper)
            else:
                organized['medium_priority'].append(paper)
        
        return organized
    
    def download_paper_batch(self, papers: List[Dict], category: str) -> None:
        """Download a batch of papers to a category folder"""
        category_dir = self.download_dir / category
        category_dir.mkdir(exist_ok=True)
        
        logger.info(f"📥 Downloading {len(papers)} papers to {category}/")
        
        for i, paper_info in enumerate(papers):
            try:
                arxiv_id = paper_info['arxiv_id']
                title = paper_info['title']
                score = paper_info.get('score', 0)
                
                # Create filename
                filename = self.create_safe_filename(title, arxiv_id)
                filepath = category_dir / filename
                
                # Skip if already exists
                if filepath.exists():
                    logger.info(f"⏭️ Already exists: {filename}")
                    continue
                
                logger.info(f"📥 [{i+1}/{len(papers)}] Downloading [{score:.1f}★]: {title[:60]}...")
                
                # Search and download
                search = arxiv.Search(id_list=[arxiv_id.split('/')[-1]])
                for paper in search.results():
                    paper.download_pdf(dirpath=str(category_dir), filename=filename)
                    
                    # Save metadata
                    metadata_file = filepath.with_suffix('.json')
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(paper_info, f, indent=2, ensure_ascii=False)
                    
                    self.downloaded_count += 1
                    logger.info(f"✅ Downloaded: {filename}")
                    break
                
                time.sleep(RATE_LIMIT_DELAY)
                
            except Exception as e:
                logger.error(f"❌ Error downloading {paper_info.get('title', 'Unknown')}: {e}")
                self.failed_count += 1
                continue
    
    def download_all_discovered_papers(self):
        """Download all papers discovered by the recursive miner"""
        logger.info("🚀 Starting Tier 1 paper download...")
        
        # Load discovered papers
        papers = self.load_discovered_papers()
        if not papers:
            return
        
        # Organize by priority
        organized = self.organize_by_category(papers)
        
        logger.info("📊 Download organization:")
        for category, paper_list in organized.items():
            logger.info(f"   {category}: {len(paper_list)} papers")
        
        start_time = time.time()
        
        # Download by priority (highest first)
        for category, paper_list in organized.items():
            if paper_list:
                logger.info(f"\n🎯 Downloading {category} papers...")
                self.download_paper_batch(paper_list, category)
        
        elapsed = time.time() - start_time
        
        # Final stats
        logger.info("🎉 " + "="*50)
        logger.info("🎉 TIER 1 PAPER DOWNLOAD COMPLETE!")
        logger.info("🎉 " + "="*50)
        logger.info(f"✅ Successfully downloaded: {self.downloaded_count}")
        logger.info(f"❌ Failed downloads: {self.failed_count}")
        logger.info(f"⏱️ Total time: {elapsed/60:.1f} minutes")
        logger.info(f"📂 Papers saved to: {self.download_dir}")
        logger.info(f"🧠 Your Tier 1 consciousness collection is ready!")
        logger.info("🎉 " + "="*50)

def main():
    """Download all discovered Tier 1 papers"""
    print("📥 TIER 1 PAPER DOWNLOADER")
    print("📚 Downloads all papers found by recursive mining")
    print("🎯 Organized by priority and source")
    print("="*50)
    
    downloader = Tier1PaperDownloader()
    downloader.download_all_discovered_papers()
    
    print("🍺 Perfect! Your consciousness paper library is complete!")

if __name__ == "__main__":
    main()
