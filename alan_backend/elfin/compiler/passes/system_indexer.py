from dataclasses import dataclass
from typing import List
from alan_backend.elfin.compiler.ast.nodes import SystemDecl

@dataclass
class SystemInfo:
    name: str
    span: "Range"   # start.line / start.col / end.line / end.col

class SystemIndexer:
    """Grab system declarations for CodeLens."""
    def run(self, ast) -> List[SystemInfo]:
        systems = []
        def _visit(n):
            if isinstance(n, SystemDecl):
                systems.append(SystemInfo(n.name, n.span))
            for child in getattr(n, "children", []):
                _visit(child)
        _visit(ast)
        return systems
