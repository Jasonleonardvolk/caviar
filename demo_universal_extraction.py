#!/usr/bin/env python3
"""
🌍 UNIVERSAL CONCEPT EXTRACTION DEMO

This script demonstrates the power of universal concept extraction
across different academic domains with real examples.
"""

import json
import sys
from pathlib import Path

# Add the ingest_pdf directory to path
sys.path.append(str(Path(__file__).parent / "ingest_pdf"))

def demo_universal_extraction():
    """Demonstrate universal concept extraction across domains"""
    
    try:
        from extractConceptsFromDocument import extractConceptsFromDocument
        print("✅ Universal extraction module loaded successfully!")
    except ImportError as e:
        print(f"❌ Failed to import extraction module: {e}")
        print("Please make sure you have run: python setup_universal_extraction.py")
        return
    
    # Test texts from different academic domains
    test_cases = [
        {
            "domain": "🧬 Quantum Physics",
            "text": """
            The coherent Ising machine utilizes degenerate optical parametric oscillators 
            to solve NP-hard optimization problems. Through x-quadrature measurement and 
            quantum superposition, the system demonstrates quantum computational advantage. 
            The Monte Carlo wave-function method provides superior modeling compared to 
            master equation approaches, particularly for non-Hermitian Hamiltonians 
            and Lindblad operator dynamics.
            """
        },
        {
            "domain": "🎭 Philosophy", 
            "text": """
            Phenomenology, as developed by Husserl, emphasizes the study of consciousness 
            and lived experience. This epistemological approach contrasts with dialectical 
            materialism's emphasis on historical processes. The hermeneutical tradition, 
            following Gadamer, focuses on interpretation and understanding, while the 
            ontological argument attempts to prove God's existence through pure reason.
            """
        },
        {
            "domain": "🎨 Art History",
            "text": """
            Chiaroscuro technique in Baroque painting creates dramatic contrasts between 
            light and shadow, as exemplified in Caravaggio's works. This influenced later 
            movements including abstract expressionism, where artists like Jackson Pollock 
            developed new approaches to color and form. The sfumato technique, perfected 
            by Leonardo da Vinci, demonstrates Renaissance mastery of atmospheric perspective.
            """
        },
        {
            "domain": "📚 Literature",
            "text": """
            Stream of consciousness narrative, pioneered by Virginia Woolf and James Joyce, 
            revolutionized modernist literature. This technique influenced magical realism 
            in Latin American literature, particularly in the works of Gabriel García Márquez. 
            Postmodernism further deconstructed traditional narrative structures, while the 
            bildungsroman continued to explore themes of personal development and growth.
            """
        },
        {
            "domain": "🧮 Mathematics",
            "text": """
            Riemannian manifolds provide the geometric foundation for general relativity, 
            while category theory offers a unifying framework for mathematical structures. 
            Galois theory reveals deep connections between field extensions and group theory, 
            and topological spaces enable the rigorous study of continuity and convergence 
            in abstract mathematical settings.
            """
        }
    ]
    
    print("🌍 UNIVERSAL CONCEPT EXTRACTION DEMO")
    print("=" * 60)
    print("Testing concept extraction across multiple academic domains...\n")
    
    all_concepts = []
    
    for i, test_case in enumerate(test_cases, 1):
        domain = test_case["domain"]
        text = test_case["text"]
        
        print(f"{domain}")
        print("-" * 40)
        
        try:
            concepts = extractConceptsFromDocument(text.strip())
            
            if concepts:
                print(f"📊 Extracted {len(concepts)} concepts:")
                
                # Show top 8 concepts
                for j, concept in enumerate(concepts[:8], 1):
                    name = concept.get('name', 'Unknown')
                    score = concept.get('score', 0)
                    method = concept.get('method', 'unknown')
                    
                    # Determine method emoji
                    method_emoji = "🔤"  # Default
                    if "yake" in method.lower():
                        method_emoji = "📈" if "keybert" in method.lower() or "ner" in method.lower() else "📊"
                    elif "keybert" in method.lower():
                        method_emoji = "🧠" if "ner" in method.lower() else "🎯"
                    elif "ner" in method.lower():
                        method_emoji = "🏷️"
                    
                    print(f"  {method_emoji} {name} (score: {score:.3f})")
                
                if len(concepts) > 8:
                    print(f"  ... and {len(concepts) - 8} more concepts")
                
                all_concepts.extend(concepts)
            else:
                print("❌ No concepts extracted")
                
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
        
        print()
    
    # Summary statistics
    if all_concepts:
        print("🎯 UNIVERSAL EXTRACTION SUMMARY")
        print("=" * 40)
        print(f"📊 Total concepts extracted: {len(all_concepts)}")
        
        # Method distribution
        methods = {}
        for concept in all_concepts:
            method = concept.get('method', 'unknown')
            if 'yake' in method.lower() and 'keybert' in method.lower():
                methods['YAKE+KeyBERT'] = methods.get('YAKE+KeyBERT', 0) + 1
            elif 'yake' in method.lower() and 'ner' in method.lower():
                methods['YAKE+NER'] = methods.get('YAKE+NER', 0) + 1
            elif 'keybert' in method.lower() and 'ner' in method.lower():
                methods['KeyBERT+NER'] = methods.get('KeyBERT+NER', 0) + 1
            elif 'yake' in method.lower():
                methods['YAKE'] = methods.get('YAKE', 0) + 1
            elif 'keybert' in method.lower():
                methods['KeyBERT'] = methods.get('KeyBERT', 0) + 1
            elif 'ner' in method.lower():
                methods['NER'] = methods.get('NER', 0) + 1
            else:
                methods['Other'] = methods.get('Other', 0) + 1
        
        print("\n🔧 Extraction method distribution:")
        for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count} concepts")
        
        # Top concepts across all domains
        sorted_concepts = sorted(all_concepts, key=lambda x: x.get('score', 0), reverse=True)
        print(f"\n🏆 Top 10 concepts across all domains:")
        for i, concept in enumerate(sorted_concepts[:10], 1):
            name = concept.get('name', 'Unknown')
            score = concept.get('score', 0)
            print(f"  {i:2}. {name} (score: {score:.3f})")
        
        print("\n✅ Universal concept extraction is working across all domains!")
        print("🌍 Ready to process papers from:")
        print("   📚 Sciences: Physics, Biology, Chemistry, Computer Science")
        print("   🎭 Humanities: Philosophy, Literature, History, Linguistics") 
        print("   🎨 Arts: Art History, Music Theory, Visual Arts")
        print("   🧮 Mathematics: Pure & Applied Mathematics")
        print("   📊 Social Sciences: Psychology, Sociology, Economics")
    else:
        print("❌ No concepts were extracted. Please check your installation.")

def demo_auto_prefill():
    """Demonstrate the auto-prefill functionality"""
    print("\n🧬 AUTO-PREFILL DATABASE DEMO")
    print("=" * 40)
    
    # Check if concept file_storages exist
    concept_db_path = Path("ingest_pdf/data/concept_file_storage.json")
    seed_db_path = Path("ingest_pdf/data/concept_seed_universal.json")
    
    if concept_db_path.exists():
        with open(concept_db_path, 'r') as f:
            main_db = json.load(f)
        print(f"📊 Main concept file_storage: {len(main_db)} concepts")
    else:
        print("⚠️ Main concept file_storage not found")
        main_db = []
    
    if seed_db_path.exists():
        with open(seed_db_path, 'r') as f:
            seed_db = json.load(f)
        print(f"🌍 Universal seed file_storage: {len(seed_db)} concepts")
        
        # Show domain distribution
        domains = {}
        for concept in seed_db:
            domain = concept.get('category', 'Unknown')
            domains[domain] = domains.get(domain, 0) + 1
        
        print("\n🌍 Seed file_storage domain coverage:")
        for domain, count in sorted(domains.items()):
            print(f"  {domain}: {count} concepts")
            
    else:
        print("⚠️ Universal seed file_storage not found")
        seed_db = []
    
    total_concepts = len(main_db) + len([s for s in seed_db if s['name'].lower() not in {c['name'].lower() for c in main_db}])
    print(f"\n📊 Total concepts available: {total_concepts}")
    print("🧬 Auto-prefill will automatically add new high-quality concepts from your documents!")

if __name__ == "__main__":
    print("🌍 UNIVERSAL CONCEPT EXTRACTION SYSTEM")
    print("🚀 Comprehensive concept extraction across ALL academic domains")
    print()
    
    demo_universal_extraction()
    demo_auto_prefill()
    
    print("\n🎯 Next Steps:")
    print("1. Install dependencies: python setup_universal_extraction.py")
    print("2. Upload a PDF through ScholarSphere to see this in action")
    print("3. Watch your concept file_storage grow automatically!")
