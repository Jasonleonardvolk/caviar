#!/usr/bin/env python
"""
PDF Manager for MCP Servers - Batch operations and management

Usage:
  python pdf_manager.py batch-add <server_name> <pdf_directory>
  python pdf_manager.py remove-pdf <server_name> <pdf_name>
  python pdf_manager.py refresh <server_name>
  python pdf_manager.py stats [server_name]
  
Examples:
  python pdf_manager.py batch-add empathy ./research/empathy_papers/
  python pdf_manager.py remove-pdf empathy old_paper.pdf
  python pdf_manager.py refresh empathy
  python pdf_manager.py stats
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import shutil

from pdf_utils import copy_and_extract_multiple, list_server_pdfs

def get_agents_dir() -> Path:
    """Get the agents directory"""
    return Path(__file__).parent.parent / "agents"

def batch_add_pdfs(server_name: str, pdf_directory: str) -> bool:
    """Add all PDFs from a directory to a server"""
    pdf_dir = Path(pdf_directory).expanduser().resolve()
    
    if not pdf_dir.exists():
        print(f"[ERROR] Directory not found: {pdf_dir}")
        return False
    
    # Find all PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    pdf_files.extend(list(pdf_dir.glob("*.PDF")))
    
    if not pdf_files:
        print(f"[WARN] No PDF files found in {pdf_dir}")
        return False
    
    print(f"[INFO] Found {len(pdf_files)} PDFs in {pdf_dir}")
    
    # Add to server
    from pdf_utils import add_pdfs_to_server
    return add_pdfs_to_server(server_name, pdf_files)

def remove_pdf(server_name: str, pdf_name: str) -> bool:
    """Remove a PDF from a server"""
    base_dir = get_agents_dir() / server_name
    spec_file = base_dir / "spec.json"
    
    if not spec_file.exists():
        print(f"[ERROR] Server '{server_name}' not found")
        return False
    
    # Load spec
    spec = json.loads(spec_file.read_text())
    original_count = len(spec["pdfs"])
    
    # Find and remove PDF
    spec["pdfs"] = [pdf for pdf in spec["pdfs"] if pdf["file"] != pdf_name]
    
    if len(spec["pdfs"]) == original_count:
        print(f"[WARN] PDF '{pdf_name}' not found in server '{server_name}'")
        return False
    
    # Update totals
    spec["total_chars"] = sum(pdf["chars"] for pdf in spec["pdfs"])
    spec["last_updated"] = datetime.utcnow().isoformat()
    
    # Save updated spec
    spec_file.write_text(json.dumps(spec, indent=2))
    
    # Remove physical file
    pdf_path = base_dir / "resources" / pdf_name
    if pdf_path.exists():
        pdf_path.unlink()
        print(f"[OK] Removed {pdf_name} from filesystem")
    
    print(f"[OK] Removed {pdf_name} from server '{server_name}'")
    return True

def refresh_server(server_name: str) -> bool:
    """Refresh server's seed.txt from all PDFs"""
    base_dir = get_agents_dir() / server_name
    spec_file = base_dir / "spec.json"
    
    if not spec_file.exists():
        print(f"[ERROR] Server '{server_name}' not found")
        return False
    
    spec = json.loads(spec_file.read_text())
    
    if not spec["pdfs"]:
        print(f"[WARN] Server '{server_name}' has no PDFs")
        return False
    
    # Re-extract text from all PDFs
    combined_text = ""
    resources_dir = base_dir / "resources"
    
    try:
        from PyPDF2 import PdfReader
        
        for pdf_info in spec["pdfs"]:
            pdf_path = resources_dir / pdf_info["file"]
            if pdf_path.exists():
                text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
                combined_text += f"\n\n--- From {pdf_info['file']} ---\n\n{text}"
                print(f"[OK] Extracted {len(text)} chars from {pdf_info['file']}")
            else:
                print(f"[WARN] PDF not found: {pdf_info['file']}")
        
        # Update seed.txt
        seed_file = base_dir / "seed.txt"
        max_chars = spec.get("seed_max_chars", 50000)
        seed_file.write_text(combined_text[:max_chars])
        
        print(f"\n[OK] Refreshed seed.txt for '{server_name}'")
        print(f"Total characters: {len(combined_text)} (truncated to {max_chars})")
        return True
        
    except ImportError:
        print("[ERROR] PyPDF2 not installed. Run: pip install PyPDF2")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to refresh: {e}")
        return False

def show_stats(server_name: Optional[str] = None) -> None:
    """Show statistics for one or all servers"""
    agents_dir = get_agents_dir()
    
    if server_name:
        # Stats for specific server
        pdfs = list_server_pdfs(server_name)
        if pdfs is None:
            print(f"[ERROR] Server '{server_name}' not found")
            return
        
        total_chars = sum(pdf["chars"] for pdf in pdfs)
        print(f"\n[Server: {server_name}]")
        print(f"PDFs: {len(pdfs)}")
        print(f"Total characters: {total_chars:,}")
        
        if pdfs:
            print("\nPDF breakdown:")
            for pdf in sorted(pdfs, key=lambda x: x["chars"], reverse=True):
                print(f"  - {pdf['file']}: {pdf['chars']:,} chars")
    
    else:
        # Stats for all servers
        print("\n[All Servers]")
        total_servers = 0
        total_pdfs = 0
        total_chars = 0
        
        for server_dir in agents_dir.iterdir():
            if server_dir.is_dir() and (server_dir / "spec.json").exists():
                spec = json.loads((server_dir / "spec.json").read_text())
                pdfs = spec.get("pdfs", [])
                if pdfs:
                    server_chars = sum(pdf["chars"] for pdf in pdfs)
                    total_servers += 1
                    total_pdfs += len(pdfs)
                    total_chars += server_chars
                    print(f"\n{server_dir.name}:")
                    print(f"  PDFs: {len(pdfs)}")
                    print(f"  Characters: {server_chars:,}")
        
        print(f"\n[Summary]")
        print(f"Servers with PDFs: {total_servers}")
        print(f"Total PDFs: {total_pdfs}")
        print(f"Total characters: {total_chars:,}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1]
    
    if command == "batch-add":
        if len(sys.argv) < 4:
            print("Usage: python pdf_manager.py batch-add <server_name> <pdf_directory>")
            return
        
        server_name = sys.argv[2]
        pdf_directory = sys.argv[3]
        batch_add_pdfs(server_name, pdf_directory)
    
    elif command == "remove-pdf":
        if len(sys.argv) < 4:
            print("Usage: python pdf_manager.py remove-pdf <server_name> <pdf_name>")
            return
        
        server_name = sys.argv[2]
        pdf_name = sys.argv[3]
        remove_pdf(server_name, pdf_name)
    
    elif command == "refresh":
        if len(sys.argv) < 3:
            print("Usage: python pdf_manager.py refresh <server_name>")
            return
        
        server_name = sys.argv[2]
        refresh_server(server_name)
    
    elif command == "stats":
        server_name = sys.argv[2] if len(sys.argv) > 2 else None
        show_stats(server_name)
    
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
