# Save as: C:\\Users\\jason\\Desktop\\tori\\kha\process_massive_datasets.py

import json
import zipfile
from pathlib import Path
import sys

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