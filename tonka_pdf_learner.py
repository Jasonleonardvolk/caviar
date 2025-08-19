# Save as: C:\\Users\\jason\\Desktop\\tori\\kha\tonka_pdf_learner.py

import os
import sys
from pathlib import Path
import json
import asyncio

# Add pigpen to path for imports
sys.path.insert(0, "C:/Users/jason/Desktop/tori/kha")

# Import your PDF ingestion system
from ingest_pdf.pipeline.pipeline import ingest_pdf_clean
from python.core import ConceptMesh

class TonkaPDFLearner:
    """
    Teach TONKA from your kick-ass PDF collection
    Uses TORI's PDF ingestion to extract knowledge
    """
    
    def __init__(self):
        self.pigpen_root = Path("C:/Users/jason/Desktop/tori/kha")
        self.pdf_library = self.pigpen_root / "pdf_library"
        self.pdf_library.mkdir(exist_ok=True)
        
        # Categories for your PDFs
        self.pdf_categories = {
            "math_physics": {
                "path": self.pdf_library / "math_physics",
                "examples": [
                    "Feynman Lectures on Physics",
                    "Spivak Calculus", 
                    "Linear Algebra Done Right",
                    "Concrete Mathematics - Knuth",
                    "Geometry - Euclid's Elements"
                ]
            },
            "programming": {
                "path": self.pdf_library / "programming",
                "examples": [
                    "SICP - Structure and Interpretation",
                    "The Art of Computer Programming - Knuth",
                    "Design Patterns - Gang of Four",
                    "Clean Code - Martin",
                    "Rust Programming Language"
                ]
            },
            "algorithms": {
                "path": self.pdf_library / "algorithms",
                "examples": [
                    "Introduction to Algorithms - CLRS",
                    "Algorithm Design Manual",
                    "Competitive Programming Handbook",
                    "The Algorithm Design Manual"
                ]
            },
            "ai_ml": {
                "path": self.pdf_library / "ai_ml",
                "examples": [
                    "Deep Learning - Goodfellow",
                    "Pattern Recognition - Bishop",
                    "The Elements of Statistical Learning",
                    "Reinforcement Learning - Sutton & Barto"
                ]
            },
            "reasoning": {
                "path": self.pdf_library / "reasoning",
                "examples": [
                    "How to Solve It - Polya",
                    "Gödel, Escher, Bach",
                    "The Art of Problem Solving",
                    "Mathematical Proofs"
                ]
            }
        }
    
    async def ingest_pdf_collection(self):
        """Ingest all PDFs using TORI's system"""
        print("📚 INGESTING YOUR KICK-ASS PDF COLLECTION")
        print("=" * 60)
        
        all_concepts = {}
        
        for category, info in self.pdf_categories.items():
            category_path = info["path"]
            if not category_path.exists():
                print(f"⚠️ Skipping {category} - no PDFs found at {category_path}")
                continue
            
            print(f"\n📖 Processing {category} PDFs...")
            category_concepts = []
            
            # Process each PDF in the category
            for pdf_file in category_path.glob("*.pdf"):
                print(f"  📄 Ingesting: {pdf_file.name}")
                
                try:
                    # Use TORI's PDF ingestion
                    result = ingest_pdf_clean(
                        str(pdf_file),
                        doc_id=pdf_file.stem,
                        extraction_threshold=0.0,
                        admin_mode=True  # Get all concepts
                    )
                    
                    # Extract concepts
                    concepts = result.get("concepts", [])
                    
                    # Tag with category and source
                    for concept in concepts:
                        concept["category"] = category
                        concept["source_pdf"] = pdf_file.name
                        concept["knowledge_type"] = self.classify_knowledge(concept, category)
                    
                    category_concepts.extend(concepts)
                    
                    print(f"    ✅ Extracted {len(concepts)} concepts")
                    
                except Exception as e:
                    print(f"    ❌ Error: {e}")
            
            all_concepts[category] = category_concepts
            print(f"  📊 Total {category} concepts: {len(category_concepts)}")
        
        # Save to TONKA's learning mesh
        self.save_to_learning_mesh(all_concepts)
        
        return all_concepts
    
    def classify_knowledge(self, concept, category):
        """Classify the type of knowledge from the concept"""
        concept_name = concept.get("name", "").lower()
        
        # Math indicators
        if any(term in concept_name for term in ["theorem", "proof", "equation", "integral", "derivative"]):
            return "mathematical"
        
        # Code indicators  
        elif any(term in concept_name for term in ["function", "class", "algorithm", "loop", "variable"]):
            return "programming"
        
        # Reasoning indicators
        elif any(term in concept_name for term in ["therefore", "implies", "conclusion", "hypothesis"]):
            return "reasoning"
        
        # Pattern indicators
        elif any(term in concept_name for term in ["pattern", "structure", "design", "architecture"]):
            return "pattern"
        
        else:
            return "general"
    
    def save_to_learning_mesh(self, all_concepts):
        """Save extracted concepts to TONKA's learning mesh"""
        print("\n💾 Saving to TONKA's learning mesh...")
        
        learning_mesh = self.pigpen_root / "concept_mesh" / "pdf_learned_concepts.json"
        learning_mesh.parent.mkdir(exist_ok=True)
        
        # Structure for learning
        structured_knowledge = {
            "math_physics": self.structure_math_concepts(all_concepts.get("math_physics", [])),
            "programming": self.structure_programming_concepts(all_concepts.get("programming", [])),
            "algorithms": self.structure_algorithm_concepts(all_concepts.get("algorithms", [])),
            "reasoning": self.structure_reasoning_concepts(all_concepts.get("reasoning", [])),
            "patterns": self.extract_cross_domain_patterns(all_concepts)
        }
        
        with open(learning_mesh, 'w', encoding='utf-8') as f:
            json.dump(structured_knowledge, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved structured knowledge to {learning_mesh}")
    
    def structure_math_concepts(self, concepts):
        """Structure mathematical concepts for learning"""
        structured = {
            "theorems": [],
            "proofs": [],
            "formulas": [],
            "methods": []
        }
        
        for concept in concepts:
            name = concept.get("name", "").lower()
            
            if "theorem" in name:
                structured["theorems"].append({
                    "name": concept["name"],
                    "statement": concept.get("description", ""),
                    "mesh_coords": concept.get("mesh_coords", [0.5, 0.5, 0.5, 0.5])
                })
            elif "proof" in name:
                structured["proofs"].append(concept)
            elif any(sym in name for sym in ["=", "∫", "∂", "Σ"]):
                structured["formulas"].append(concept)
            else:
                structured["methods"].append(concept)
        
        return structured
    
    def structure_programming_concepts(self, concepts):
        """Structure programming concepts for code generation"""
        structured = {
            "design_patterns": [],
            "algorithms": [],
            "data_structures": [],
            "best_practices": []
        }
        
        for concept in concepts:
            name = concept.get("name", "").lower()
            
            if "pattern" in name:
                structured["design_patterns"].append(concept)
            elif "algorithm" in name or "sort" in name or "search" in name:
                structured["algorithms"].append(concept)
            elif any(ds in name for ds in ["tree", "graph", "list", "queue", "stack"]):
                structured["data_structures"].append(concept)
            else:
                structured["best_practices"].append(concept)
        
        return structured
    
    def structure_algorithm_concepts(self, concepts):
        """Structure algorithm concepts"""
        return {
            "sorting": [c for c in concepts if "sort" in c.get("name", "").lower()],
            "searching": [c for c in concepts if "search" in c.get("name", "").lower()],
            "graph": [c for c in concepts if "graph" in c.get("name", "").lower()],
            "dynamic": [c for c in concepts if "dynamic" in c.get("name", "").lower()],
            "greedy": [c for c in concepts if "greedy" in c.get("name", "").lower()]
        }
    
    def structure_reasoning_concepts(self, concepts):
        """Structure reasoning patterns"""
        return {
            "problem_solving": [c for c in concepts if "solve" in c.get("name", "").lower()],
            "logical": [c for c in concepts if any(term in c.get("name", "").lower() 
                                                  for term in ["logic", "proof", "theorem"])],
            "heuristics": [c for c in concepts if "heuristic" in c.get("name", "").lower()]
        }
    
    def extract_cross_domain_patterns(self, all_concepts):
        """Extract patterns that appear across domains"""
        patterns = {}
        
        # Look for common patterns across categories
        all_concept_names = []
        for category_concepts in all_concepts.values():
            all_concept_names.extend([c.get("name", "") for c in category_concepts])
        
        # Find recurring themes
        from collections import Counter
        theme_counts = Counter(all_concept_names)
        
        patterns["recurring_themes"] = [
            {"theme": theme, "count": count} 
            for theme, count in theme_counts.most_common(20)
        ]
        
        return patterns
    
    def quick_pdf_test(self):
        """Quick test with a single PDF"""
        print("\n🧪 Quick PDF Learning Test")
        print("=" * 40)
        
        # Find first PDF
        for category, info in self.pdf_categories.items():
            pdf_path = info["path"]
            if pdf_path.exists():
                pdfs = list(pdf_path.glob("*.pdf"))
                if pdfs:
                    test_pdf = pdfs[0]
                    print(f"Testing with: {test_pdf.name}")
                    
                    result = ingest_pdf_clean(str(test_pdf))
                    concepts = result.get("concepts", [])
                    
                    print(f"✅ Extracted {len(concepts)} concepts")
                    if concepts:
                        print("\nSample concepts:")
                        for c in concepts[:5]:
                            print(f"  - {c.get('name')}: {c.get('score', 0):.2f}")
                    
                    return True
        
        print("❌ No PDFs found for testing")
        return False

async def main():
    """Process your PDF collection"""
    learner = TonkaPDFLearner()
    
    print("📚 TONKA PDF LEARNING SYSTEM")
    print("Teaching TONKA from your PDF collection")
    print("=" * 60)
    
    # First, organize your PDFs
    print("\n📁 Expected PDF structure:")
    for category, info in learner.pdf_categories.items():
        print(f"  {info['path']}/")
        for example in info["examples"][:3]:
            print(f"    - {example}.pdf")
    
    print("\n1️⃣ Place your PDFs in the appropriate folders")
    print("2️⃣ Press Enter when ready...")
    input()
    
    # Quick test
    if learner.quick_pdf_test():
        print("\n✅ PDF ingestion working!")
        
        # Full ingestion
        print("\n🚀 Starting full PDF collection ingestion...")
        concepts = await learner.ingest_pdf_collection()
        
        # Summary
        total = sum(len(c) for c in concepts.values())
        print(f"\n📊 LEARNING COMPLETE!")
        print(f"Total concepts extracted: {total}")
        for category, cat_concepts in concepts.items():
            print(f"  {category}: {len(cat_concepts)} concepts")
    
    print("\n✅ TONKA has learned from your PDFs!")
    print("🚀 Next: Test TONKA with questions from your PDFs")

if __name__ == "__main__":
    asyncio.run(main())