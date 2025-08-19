#!/usr/bin/env python3
"""
Dependency-graph generator
==========================
Builds an import/require graph for:

* Python (.py) –  `import x`, `from x import …`
* JS/TS (.js/.ts) –  `require("x")`, `import … from "x"`
* Elfin (.elfin) –  naive `use x` statements

Outputs `deps_current.md` with a hierarchical tree and any cycles.

Usage
-----
    python tools/deps_tree.py
    python tools/deps_tree.py --root kha --ext .py,.js,.ts,.elfin
    python tools/deps_tree.py --root . --skip-dir node_modules
"""

import argparse
import os
import pathlib
import re
import sys
import textwrap
from typing import Iterable, Set, List
import networkx as nx

# Directories to skip by default
SKIP_DIRS: Set[str] = {'.git', '.venv', 'node_modules', '__pycache__', 'dist', 'build'}

# Regex patterns for different file types
PY_RE   = re.compile(r"^\s*(?:from|import)\s+([\w\.]+)", re.M)
JS_RE   = re.compile(
    r'require\([\'"]([\w\./-]+)[\'"]\)|import .* from [\'"]([\w\./-]+)[\'"]', re.M
)
ELF_RE  = re.compile(r"\buse\s+([\w\.]+)", re.M)

def iter_files(root: pathlib.Path, exts: Set[str]) -> Iterable[pathlib.Path]:
    """Yield every file under root with matching extensions, skipping problematic dirs."""
    for dpath, dnames, fnames in os.walk(root, onerror=lambda e: None, followlinks=False):
        # Filter out directories we want to skip
        dnames[:] = [d for d in dnames if d not in SKIP_DIRS]
        for fname in fnames:
            fpath = pathlib.Path(dpath) / fname
            if fpath.suffix.lower() in exts:
                yield fpath

def _deps(path: pathlib.Path) -> List[str]:
    """Extract dependencies from a file based on its type."""
    try:
        txt = path.read_text(errors="ignore")
    except (FileNotFoundError, PermissionError, OSError):
        return []
    
    if path.suffix == ".py":
        return PY_RE.findall(txt)
    if path.suffix in {".js", ".ts"}:
        return [m for pair in JS_RE.findall(txt) for m in pair if m]
    if path.suffix == ".elfin":
        return ELF_RE.findall(txt)
    return []

def build_graph(root: pathlib.Path, exts: Set[str]) -> nx.DiGraph:
    """Build a directed graph of dependencies."""
    G = nx.DiGraph()
    
    for p in iter_files(root, exts):
        try:
            mod = p.relative_to(root).as_posix()
        except ValueError:
            # If the path is not relative to root, use the full path
            mod = p.as_posix()
        
        G.add_node(mod)
        for dep in _deps(p):
            G.add_edge(mod, dep)
    
    return G

def markdown_dump(G: nx.DiGraph, out: pathlib.Path):
    """Write the dependency graph to a markdown file."""
    with out.open("w", encoding="utf-8") as f:
        f.write("# Dependency Tree\n\n")
        f.write(f"Total modules: {G.number_of_nodes()}\n")
        f.write(f"Total dependencies: {G.number_of_edges()}\n\n")
        
        # Try topological sort, fallback to regular nodes if cyclic
        try:
            order = list(nx.topological_sort(G))
            f.write("## Dependency Hierarchy\n\n")
        except nx.NetworkXUnfeasible:
            order = list(G.nodes())
            f.write("## Dependency Hierarchy (Warning: cycles detected)\n\n")

        # Group nodes by their depth in the dependency tree
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        # Write tree structure
        visited = set()
        
        def write_node(node, indent=0):
            if node in visited:
                return
            visited.add(node)
            
            prefix = "  " * indent + "- "
            f.write(f"{prefix}{node}\n")
            
            # Write dependencies of this node
            for child in sorted(G.successors(node)):
                if child not in visited:
                    write_node(child, indent + 1)
        
        # Start with root nodes
        for root in sorted(roots):
            write_node(root)
        
        # Write any remaining nodes not reached from roots
        remaining = set(G.nodes()) - visited
        if remaining:
            f.write("\n## Isolated Modules\n\n")
            for node in sorted(remaining):
                f.write(f"- {node}\n")

        # Detect and report cycles
        cycles = list(nx.simple_cycles(G))
        if cycles:
            f.write("\n## Cycles Detected\n\n")
            for i, cycle in enumerate(cycles, 1):
                f.write(f"{i}. {' → '.join(cycle)} → {cycle[0]}\n")

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate import/require dependency graph.")
    ap.add_argument("--root", default=".", help="Project root directory")
    ap.add_argument(
        "--ext", 
        default=".py,.js,.ts,.elfin", 
        help="Comma-separated file extensions to analyze"
    )
    ap.add_argument(
        "--skip-dir", 
        action='append', 
        default=[],
        help='Directory name to ignore (can be given multiple times)'
    )
    ap.add_argument(
        "--output", "-o",
        default="deps_current.md",
        help="Output markdown file name"
    )
    ap.add_argument(
        "--verbose", "-v",
        action='store_true',
        help='Show more detailed output'
    )
    args = ap.parse_args()

    # Update SKIP_DIRS with any additional directories from command line
    if args.skip_dir:
        SKIP_DIRS.update(args.skip_dir)

    root = pathlib.Path(args.root).resolve()
    exts = {e.strip().lower() for e in args.ext.split(",") if e.strip()}

    if not root.exists():
        sys.exit(f"[deps_tree] Root path does not exist: {root}")

    if args.verbose:
        print(f"[deps_tree] Scanning: {root}")
        print(f"[deps_tree] Extensions: {', '.join(sorted(exts))}")
        print(f"[deps_tree] Skipping dirs: {', '.join(sorted(SKIP_DIRS))}")

    G = build_graph(root, exts)
    
    output_path = pathlib.Path(args.output)
    markdown_dump(G, output_path)
    
    print(f"[deps_tree] {output_path} written.")
    print(f"[deps_tree] Found {G.number_of_nodes()} modules with {G.number_of_edges()} dependencies.")
    
    cycles = list(nx.simple_cycles(G))
    if cycles:
        print(f"[deps_tree] Warning: {len(cycles)} circular dependencies detected!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
