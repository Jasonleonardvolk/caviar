# Save as: C:\\Users\\jason\\Desktop\\tori\\kha\process_massive_datasets_fixed.py

import json
import zipfile
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, "C:/Users/jason/Desktop/tori/kha")

class MassiveDatasetProcessor:
    """Process all the massive datasets for TONKA"""
    
    def __init__(self):
        self.dataset_dir = Path("C:/Users/jason/Desktop/tori/kha/docs/material/dataset/06_27")
        self.concept_mesh_dir = Path("C:/Users/jason/Desktop/tori/kha/concept_mesh")
        self.concept_mesh_dir.mkdir(exist_ok=True)
        
        self.all_concepts = {
            "programming": [],
            "mathematics": [],
            "algorithms": [],
            "patterns": [],
            "reasoning": [],
            "problems": []
        }
        
    def process_all_datasets(self):
        """Process all downloaded datasets"""
        print("🚀 PROCESSING MASSIVE DATASETS FOR TONKA")
        print("=" * 60)
        
        # 1. Process full MBPP
        print("\n📚 Processing 974 MBPP Problems...")
        self.process_mbpp_complete()
        
        # 2. Process HumanEval
        print("\n📚 Processing HumanEval...")
        self.process_humaneval()
        
        # 3. Process Project Euler
        print("\n📚 Processing Project Euler...")
        self.process_project_euler()
        
        # 4. Process Rosetta Code
        print("\n📚 Processing Rosetta Code...")
        self.process_rosetta_code()
        
        # 5. Process Advanced Mathematics
        print("\n📚 Processing Advanced Mathematics...")
        self.process_advanced_math()
        
        # 6. Process Algorithm Manual
        print("\n📚 Processing Algorithm Design Manual...")
        self.process_algorithms()
        
        # 7. Process Real-world Patterns
        print("\n📚 Processing Real-world Patterns...")
        self.process_real_world()
        
        # Save everything
        self.save_to_concept_mesh()
        
        # Create summary
        self.create_processing_summary()
    
    def process_mbpp_complete(self):
        """Process all MBPP problems"""
        mbpp_file = self.dataset_dir / "mbpp_complete.json"
        if not mbpp_file.exists():
            print("❌ MBPP complete file not found")
            return
            
        with open(mbpp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        problems = data.get("problems", [])
        print(f"   Processing {len(problems)} MBPP problems...")
        
        for i, problem in enumerate(problems):
            concept = {
                "id": f"mbpp_{i}",
                "name": f"MBPP Problem {i}",
                "task": problem.get("text", ""),
                "solution": problem.get("code", ""),
                "tests": problem.get("test_list", []),
                "type": "programming_problem",
                "language": "python",
                "source": "MBPP",
                "difficulty": self.estimate_difficulty(problem),
                "mesh_coords": self.calculate_coords("mbpp", i)
            }
            self.all_concepts["programming"].append(concept)
        
        print(f"   ✅ Processed {len(problems)} MBPP problems")
    
    def process_humaneval(self):
        """Process HumanEval dataset"""
        humaneval_zip = self.dataset_dir / "humaneval.zip"
        if humaneval_zip.exists():
            try:
                with zipfile.ZipFile(humaneval_zip, 'r') as zip_ref:
                    # Extract to temp dir
                    extract_dir = self.dataset_dir / "humaneval_temp"
                    zip_ref.extractall(extract_dir)
                    
                    # Find the data file
                    for file in extract_dir.rglob("*.jsonl"):
                        with open(file, 'r', encoding='utf-8') as f:
                            count = 0
                            for line in f:
                                try:
                                    problem = json.loads(line)
                                    concept = {
                                        "id": f"humaneval_{count}",
                                        "name": problem.get("task_id", f"HumanEval_{count}"),
                                        "prompt": problem.get("prompt", ""),
                                        "canonical_solution": problem.get("canonical_solution", ""),
                                        "test": problem.get("test", ""),
                                        "type": "programming_challenge",
                                        "language": "python",
                                        "source": "HumanEval",
                                        "mesh_coords": self.calculate_coords("humaneval", count)
                                    }
                                    self.all_concepts["programming"].append(concept)
                                    count += 1
                                except:
                                    continue
                            print(f"   ✅ Processed {count} HumanEval problems")
                            break
            except Exception as e:
                print(f"   ⚠️ Error processing HumanEval: {e}")
    
    def process_project_euler(self):
        """Process Project Euler problems"""
        euler_file = self.dataset_dir / "project_euler.json"
        if not euler_file.exists():
            return
            
        with open(euler_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        problems = data.get("problems", [])
        for problem in problems:
            concept = {
                "id": f"euler_{problem['id']}",
                "name": problem.get("title", f"Problem {problem['id']}"),
                "description": problem.get("description", ""),
                "solution": problem.get("solution_python", ""),
                "answer": problem.get("answer", None),
                "type": "mathematical_problem",
                "tags": problem.get("tags", ["mathematics", "algorithms"]),
                "difficulty": problem.get("difficulty", "medium"),
                "source": "Project_Euler",
                "mesh_coords": self.calculate_coords("euler", problem['id'])
            }
            self.all_concepts["problems"].append(concept)
        
        print(f"   ✅ Processed {len(problems)} Project Euler problems")
    
    def process_rosetta_code(self):
        """Process Rosetta Code examples"""
        rosetta_file = self.dataset_dir / "rosetta_code.json"
        if not rosetta_file.exists():
            return
            
        with open(rosetta_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for category, tasks in data.get("tasks", {}).items():
            for task_name, implementations in tasks.items():
                if isinstance(implementations, dict):
                    concept = {
                        "id": f"rosetta_{category}_{task_name}",
                        "name": task_name,
                        "category": category,
                        "implementations": {},
                        "type": "multi_language_pattern",
                        "source": "Rosetta_Code",
                        "mesh_coords": self.calculate_coords("rosetta", f"{category}_{task_name}")
                    }
                    
                    # Extract implementations by language
                    for lang, code in implementations.items():
                        if lang not in ["description", "complexity", "operations", "applications"]:
                            concept["implementations"][lang] = code
                    
                    if concept["implementations"]:
                        self.all_concepts["patterns"].append(concept)
                        count += 1
        
        print(f"   ✅ Processed {count} Rosetta Code patterns")
    
    def process_advanced_math(self):
        """Process advanced mathematics"""
        math_file = self.dataset_dir / "advanced_mathematics.json"
        if not math_file.exists():
            return
            
        with open(math_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for subject, topics in data.items():
            for topic_name, content in topics.items():
                if isinstance(content, list):
                    # Process list of examples
                    for item in content:
                        concept = {
                            "id": f"math_{subject}_{topic_name}_{count}",
                            "name": item.get("function", item.get("name", topic_name)),
                            "subject": subject,
                            "topic": topic_name,
                            "formula": item.get("derivative", item.get("integral", "")),
                            "code": item.get("code", ""),
                            "type": "mathematical_concept",
                            "source": "Advanced_Math",
                            "mesh_coords": self.calculate_coords("math", f"{subject}_{count}")
                        }
                        self.all_concepts["mathematics"].append(concept)
                        count += 1
                elif isinstance(content, dict):
                    # Process structured content
                    for subtopic, details in content.items():
                        if isinstance(details, str) and ("def " in details or "lambda" in details):
                            concept = {
                                "id": f"math_{subject}_{topic_name}_{subtopic}",
                                "name": subtopic,
                                "subject": subject,
                                "topic": topic_name,
                                "implementation": details,
                                "type": "mathematical_algorithm",
                                "source": "Advanced_Math",
                                "mesh_coords": self.calculate_coords("math", f"{subject}_{subtopic}")
                            }
                            self.all_concepts["mathematics"].append(concept)
                            count += 1
        
        print(f"   ✅ Processed {count} mathematical concepts")
    
    def process_algorithms(self):
        """Process algorithm manual"""
        algo_file = self.dataset_dir / "algorithm_design_manual.json"
        if not algo_file.exists():
            return
            
        with open(algo_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for category, algorithms in data.items():
            for algo_name, details in algorithms.items():
                if isinstance(details, dict):
                    concept = {
                        "id": f"algo_{category}_{algo_name}",
                        "name": algo_name,
                        "category": category,
                        "description": details.get("description", ""),
                        "implementation": details.get("implementation", ""),
                        "complexity": details.get("complexity", ""),
                        "applications": details.get("applications", []),
                        "type": "algorithm",
                        "source": "Algorithm_Manual",
                        "mesh_coords": self.calculate_coords("algo", f"{category}_{algo_name}")
                    }
                    
                    # Handle nested implementations
                    if not concept["implementation"] and isinstance(details, dict):
                        for key, value in details.items():
                            if isinstance(value, str) and "def " in value:
                                concept["implementation"] = value
                                break
                    
                    self.all_concepts["algorithms"].append(concept)
                    count += 1
        
        print(f"   ✅ Processed {count} algorithms")
    
    def process_real_world(self):
        """Process real-world patterns"""
        patterns_file = self.dataset_dir / "real_world_patterns.json"
        if not patterns_file.exists():
            return
            
        with open(patterns_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for category, patterns in data.items():
            for pattern_name, details in patterns.items():
                if isinstance(details, dict):
                    concept = {
                        "id": f"real_{category}_{pattern_name}",
                        "name": pattern_name,
                        "category": category,
                        "description": details.get("description", ""),
                        "implementation": details.get("implementation", details.get("fastapi", details.get("example", ""))),
                        "type": "real_world_pattern",
                        "source": "Real_World",
                        "mesh_coords": self.calculate_coords("real", f"{category}_{pattern_name}")
                    }
                    self.all_concepts["patterns"].append(concept)
                    count += 1
        
        print(f"   ✅ Processed {count} real-world patterns")
    
    def estimate_difficulty(self, problem):
        """Estimate problem difficulty based on content"""
        code = problem.get("code", "")
        
        # Simple heuristics
        if len(code) < 50:
            return "easy"
        elif len(code) < 150:
            return "medium"
        else:
            return "hard"
    
    def calculate_coords(self, prefix, identifier):
        """Calculate 4D mesh coordinates"""
        h1 = hash(prefix)
        h2 = hash(str(identifier))
        
        return [
            (h1 % 1000) / 1000,
            (h2 % 1000) / 1000,
            ((h1 + h2) % 1000) / 1000,
            ((h1 * h2) % 1000) / 1000
        ]
    
    def save_to_concept_mesh(self):
        """Save all processed concepts"""
        print("\n💾 Saving to Concept Mesh...")
        
        # Calculate totals
        total_concepts = sum(len(concepts) for concepts in self.all_concepts.values())
        
        # Save main file
        mesh_data = {
            "metadata": {
                "version": "2.0",
                "source": "massive_datasets",
                "total_concepts": total_concepts,
                "processing_date": datetime.now().isoformat()
            },
            "concepts": self.all_concepts,
            "statistics": {
                category: len(concepts) 
                for category, concepts in self.all_concepts.items()
            }
        }
        
        main_file = self.concept_mesh_dir / "tonka_massive_concepts.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(mesh_data, f, indent=2)
        
        print(f"   ✅ Saved {total_concepts} total concepts")
        
        # Save category files
        for category, concepts in self.all_concepts.items():
            if concepts:
                category_file = self.concept_mesh_dir / f"massive_{category}.json"
                with open(category_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "category": category,
                        "count": len(concepts),
                        "concepts": concepts
                    }, f, indent=2)
                print(f"   ✅ Saved {len(concepts)} {category} concepts")
    
    def create_processing_summary(self):
        """Create final summary"""
        summary = {
            "processing_complete": True,
            "total_concepts": sum(len(c) for c in self.all_concepts.values()),
            "breakdown": {
                category: len(concepts)
                for category, concepts in self.all_concepts.items()
            },
            "ready_for_tonka": True,
            "next_steps": [
                "Run TONKA MCP server",
                "Test with complex coding tasks",
                "TONKA can now code like a badass!"
            ]
        }
        
        summary_file = self.concept_mesh_dir / "PROCESSING_COMPLETE.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("🎉 MASSIVE DATASET PROCESSING COMPLETE!")
        print(f"📊 Total concepts processed: {summary['total_concepts']}")
        for category, count in summary['breakdown'].items():
            print(f"   - {category}: {count} concepts")
        print("\n✅ TONKA IS NOW A CODING BADASS!")

def main():
    processor = MassiveDatasetProcessor()
    processor.process_all_datasets()
    
    print("\n🚀 Next: Create TONKA MCP Server")
    print("Run: C:\\ALANPY311\\python.exe create_tonka_mcp.py")

if __name__ == "__main__":
    main()