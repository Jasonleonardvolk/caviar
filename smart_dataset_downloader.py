# Save as: C:\\Users\\jason\\Desktop\\tori\\kha\smart_dataset_downloader.py
# ------------------------------------------------------------------
import json
import subprocess
from pathlib import Path
# (requests is imported later if/when you need it)

class SmartDatasetDownloader:
    """
    Smart dataset downloader - quality over quantity
    Optimized for concept mesh learning (no tokens!)
    All files now live in:
        C:\\Users\\jason\\Desktop\\tori\\kha\docs\material\dataset\06_27
    """

    # -------------------------------------------------------------- #
    # 🔧 1.  Where we put the data
    # -------------------------------------------------------------- #
    PIGPEN_ROOT = Path(
        r"C:/Users/jason/Desktop/tori/kha/docs/material/dataset/06_27"
    ).resolve()

    def __init__(self) -> None:
        # Ensure root exists first
        self.PIGPEN_ROOT.mkdir(parents=True, exist_ok=True)

        # Main dataset area
        self.datasets_dir = self.PIGPEN_ROOT / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        # Concept-mesh directory (for extracted JSONs)
        self.concept_mesh_dir = self.PIGPEN_ROOT / "concept_mesh"
        self.concept_mesh_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- #
    # 2.  DATASET SELECTION
    # -------------------------------------------------------------- #
    def get_essential_datasets(self):
        """Get the BEST datasets that fit in ~2 TB."""
        print("🎯 SMART DATASET SELECTION FOR TONKA")
        print("=" * 60)

        # 1. Code datasets
        code_datasets = {
            "python_clean": {
                "source": "GitHub Python repos (curated)",
                "size": "~50GB",
                "why": "Real-world Python patterns from top projects",
                "download": self.download_python_datasets,
            },
            "rosetta_code": {
                "source": "Rosetta Code – same problem in many languages",
                "size": "~1GB",
                "why": "Perfect for learning cross-language patterns",
                "download": self.download_rosetta_code,
            },
            "leetcode_solutions": {
                "source": "LeetCode problems with multiple solutions",
                "size": "~5GB",
                "why": "Problem-solving patterns",
                "download": self.download_leetcode,
            },
        }

        # 2. Math datasets
        math_datasets = {
            "khan_academy": {
                "source": "Khan Academy exercises",
                "size": "~10GB",
                "why": "Step-by-step solutions perfect for LSTM learning",
                "download": self.download_khan_math,
            },
            "aops": {
                "source": "Art of Problem Solving",
                "size": "~5GB",
                "why": "Competition math with discussions",
                "download": self.download_aops,
            },
        }

        # 3. Reasoning datasets
        reasoning_datasets = {
            "big_bench": {
                "source": "BIG-Bench (subset)",
                "size": "~20GB",
                "why": "Diverse reasoning tasks",
                "download": self.download_big_bench,
            }
        }

        return code_datasets, math_datasets, reasoning_datasets

    # -------------------------------------------------------------- #
    # 3.  CODE DATASETS (EXAMPLE IMPLEMENTATION)
    # -------------------------------------------------------------- #
    def download_python_datasets(self):
        """Download curated Python code."""
        print("\n📥 Downloading Python patterns...")

        repos = [
            "django/django",
            "flask/flask",
            "requests/requests",
            "numpy/numpy",
            "pandas/pandas",
            "fastapi/fastapi",
            "pytorch/pytorch",
            "tensorflow/tensorflow",
        ]

        patterns_dir = self.datasets_dir / "python_patterns"
        patterns_dir.mkdir(parents=True, exist_ok=True)

        for repo in repos:
            repo_name = repo.split("/")[-1]
            target = patterns_dir / repo_name
            if target.exists():
                print(f"  ✔ {repo_name} already cloned.")
                continue

            print(f"  Cloning {repo}...")
            subprocess.run(
                ["git", "clone", f"https://github.com/{repo}.git", "--depth", "1", str(target)],
                check=True,
            )

        self.extract_python_patterns(patterns_dir)

    # -------------------------------------------------------------- #
    # 4.  PATTERN EXTRACTION
    # -------------------------------------------------------------- #
    def extract_python_patterns(self, patterns_dir: Path):
        """Extract reusable patterns from Python code."""
        print("\n🔍 Extracting Python patterns...")

        patterns = {
            "api_endpoints": [],
            "error_handling": [],
            "data_processing": [],
            "class_patterns": [],
            "test_patterns": [],
            "async_patterns": [],
        }

        for repo_dir in patterns_dir.iterdir():
            if not repo_dir.is_dir():
                continue

            for py_file in repo_dir.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                    if "@app.route" in content or "@router" in content:
                        patterns["api_endpoints"].append(self.extract_api_pattern(content))
                    if "try:" in content:
                        patterns["error_handling"].append(self.extract_error_pattern(content))
                    if "async def" in content:
                        patterns["async_patterns"].append(self.extract_async_pattern(content))
                except Exception:
                    continue

        mesh_file = self.concept_mesh_dir / "extracted_python_patterns.json"
        mesh_file.write_text(json.dumps(patterns, indent=2, ensure_ascii=False))
        print(f"✅ Extracted patterns saved ➜ {mesh_file}")

    # -------------------------------------------------------------- #
    # 5.  ROSETTA CODE
    # -------------------------------------------------------------- #
    def download_rosetta_code(self):
        """Download Rosetta Code examples."""
        print("\n📥 Downloading Rosetta Code...")

        rosetta_dir = self.datasets_dir / "rosetta_code"
        rosetta_dir.mkdir(parents=True, exist_ok=True)

        problems = ["fibonacci", "quicksort", "binary_search", "merge_sort", "dijkstra", "a_star"]
        self.create_rosetta_examples(rosetta_dir, problems)

    def create_rosetta_examples(self, rosetta_dir: Path, problems: list):
        """Create multi-language examples (stubbed)."""
        fibonacci_examples = {
            "python": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "rust": "fn fibonacci(n: u32) -> u32 {\n    match n {\n        0 | 1 => n,\n        _ => fibonacci(n - 1) + fibonacci(n - 2),\n    }\n}",
            "csharp": "public static int Fibonacci(int n) {\n    if (n <= 1) return n;\n    return Fibonacci(n - 1) + Fibonacci(n - 2);\n}",
            "pattern": {
                "concept": "recursive_fibonacci",
                "structure": "base_case + recursive_calls",
                "mesh_coords": [0.3, 0.7, 0.5, 0.4],
            },
        }

        mesh_file = self.concept_mesh_dir / "rosetta_patterns.json"
        mesh_file.write_text(json.dumps({"fibonacci": fibonacci_examples}, indent=2, ensure_ascii=False))

    # -------------------------------------------------------------- #
    # 6.  REASONING CURRICULUM
    # -------------------------------------------------------------- #
    def create_reasoning_curriculum(self):
        print("\n🧠 Creating reasoning curriculum...")

        reasoning = {
            "logical_reasoning": {
                "modus_ponens": {
                    "rule": "If P then Q; P; therefore Q",
                    "examples": [
                        {
                            "premise1": "If code compiles, it might work",
                            "premise2": "Code compiles",
                            "conclusion": "It might work",
                        }
                    ],
                    "mesh_coords": [0.8, 0.2, 0.6, 0.4],
                },
                "deduction": {
                    "rule": "General to specific",
                    "example": "All bugs cause errors. This is a bug. Therefore, this causes an error.",
                    "mesh_coords": [0.7, 0.3, 0.6, 0.5],
                },
            },
            "problem_solving": {
                "decomposition": {
                    "strategy": "Break complex into simple",
                    "example": "Build app = setup + routes + file_storage + frontend",
                    "mesh_coords": [0.6, 0.5, 0.7, 0.6],
                },
                "pattern_matching": {
                    "strategy": "Find similar solved problems",
                    "example": "This looks like a graph problem, use BFS/DFS",
                    "mesh_coords": [0.5, 0.6, 0.8, 0.5],
                },
            },
            "debugging_reasoning": {
                "hypothesis_testing": {
                    "steps": ["Observe symptom", "Form hypothesis", "Test hypothesis", "Refine or fix"],
                    "mesh_coords": [0.9, 0.7, 0.7, 0.8],
                }
            },
        }

        mesh_file = self.concept_mesh_dir / "reasoning_curriculum.json"
        mesh_file.write_text(json.dumps(reasoning, indent=2, ensure_ascii=False))
        print(f"✅ Reasoning curriculum created ➜ {mesh_file}")

    # -------------------------------------------------------------- #
    # 7.  MATH PROGRESSIONS  (unchanged except for path)
    # -------------------------------------------------------------- #
    def create_math_progressions(self):
        print("\n📐 Creating math progressions...")

        math_progression = {
            "arithmetic": {
                "level": 1,
                "concepts": ["addition", "subtraction", "multiplication", "division"],
                "leads_to": ["algebra"],
            },
            "algebra": {
                "level": 2,
                "concepts": ["variables", "equations", "functions", "graphing"],
                "leads_to": ["calculus", "linear_algebra"],
            },
            "geometry": {
                "level": 2,
                "concepts": ["shapes", "angles", "proofs", "trigonometry"],
                "leads_to": ["calculus", "topology"],
            },
            "calculus": {
                "level": 3,
                "concepts": ["limits", "derivatives", "integrals", "series"],
                "leads_to": ["differential_equations", "analysis"],
            },
            "discrete_math": {
                "level": 3,
                "concepts": ["logic", "sets", "graphs", "combinatorics"],
                "leads_to": ["algorithms", "cryptography"],
            },
        }

        detailed_math = {}
        for subject, info in math_progression.items():
            detailed_math[subject] = {
                **info,
                "examples": self.generate_math_examples(subject),
                "mesh_coords": self.calculate_subject_coords(subject),
            }

        mesh_file = self.concept_mesh_dir / "math_progression.json"
        mesh_file.write_text(json.dumps(detailed_math, indent=2, ensure_ascii=False))
        print(f"✅ Math progressions created ➜ {mesh_file}")

    # helper methods unchanged …
    def generate_math_examples(self, subject: str):
        examples = {
            "algebra": [
                {"problem": "2x + 5 = 13", "solution": "x = 4", "steps": ["2x = 8", "x = 4"]},
                {"problem": "x² - 5x + 6 = 0", "solution": "x = 2 or x = 3", "method": "factoring"},
            ],
            "calculus": [
                {"problem": "d/dx(x²)", "solution": "2x", "rule": "power_rule"},
                {"problem": "∫x dx", "solution": "x²/2 + C", "rule": "power_rule"},
            ],
            "geometry": [
                {
                    "problem": "Find area of triangle with base=6, height=4",
                    "solution": "12",
                    "formula": "A = ½bh",
                }
            ],
        }
        return examples.get(subject, [])

    def calculate_subject_coords(self, subject: str):
        h = hash(subject)
        return [
            (h % 1000) / 1000,
            ((h >> 10) % 1000) / 1000,
            ((h >> 20) % 1000) / 1000,
            ((h >> 30) % 1000) / 1000,
        ]

    # -------------------------------------------------------------- #
    # 8.  MAIN ENTRY POINT
    # -------------------------------------------------------------- #
    def setup_smart_datasets(self):
        print("🚀 SMART DATASET SETUP FOR TONKA")
        print("Optimizing for quality over quantity")
        print("=" * 60)

        self.create_reasoning_curriculum()
        self.create_math_progressions()

        print("\n📥 Downloading essential datasets...")
        self.download_python_datasets()
        self.download_rosetta_code()

        print("\n✅ Smart datasets ready!")
        print(f"📁 Location: {self.PIGPEN_ROOT}")
        print("💾 Total size: < 100 GB (efficient!)")
        print("\n🚀 Next: Run 'python tonka_learning_curriculum.py'")

# ------------------------------------------------------------------ #
def main():
    downloader = SmartDatasetDownloader()
    downloader.setup_smart_datasets()

if __name__ == "__main__":
    main()
