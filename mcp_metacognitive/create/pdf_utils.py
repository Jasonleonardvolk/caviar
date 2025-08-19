import shutil, hashlib, json
from pathlib import Path
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Optional
from datetime import datetime

TMP = Path(__file__).parent.parent / "_resources"

def copy_and_extract(pdf_path: Path, dest_dir: Path) -> dict:
    """Copy PDF into agent/resources, return extracted text + md5."""
    dest_resources = dest_dir / "resources"
    dest_resources.mkdir(parents=True, exist_ok=True)
    pdf_copy = dest_resources / pdf_path.name
    shutil.copy2(pdf_path, pdf_copy)

    text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_copy).pages)
    md5  = hashlib.md5(pdf_copy.read_bytes()).hexdigest()
    meta = {"file": pdf_copy.name, "md5": md5, "chars": len(text)}
    return meta, text

def copy_and_extract_multiple(pdf_paths: List[Path], dest_dir: Path, append: bool = False) -> Dict[str, Any]:
    """
    Copy multiple PDFs into agent/resources, extract and combine texts.
    
    Args:
        pdf_paths: List of PDF paths to process
        dest_dir: Destination directory for the agent
        append: If True, append to existing spec.json instead of overwriting
    
    Returns:
        Combined metadata for all PDFs
    """
    dest_resources = dest_dir / "resources"
    dest_resources.mkdir(parents=True, exist_ok=True)
    
    # Load existing spec if appending
    spec_file = dest_dir / "spec.json"
    if append and spec_file.exists():
        existing_spec = json.loads(spec_file.read_text())
    else:
        existing_spec = {
            "pdfs": [],
            "total_chars": 0,
            "last_updated": None,
            "seed_max_chars": 50000
        }
    
    # Process each PDF
    combined_text = ""
    new_pdfs = []
    
    for pdf_path in pdf_paths:
        try:
            # Skip if already processed (by checking MD5)
            pdf_copy = dest_resources / pdf_path.name
            
            # If appending and file exists, check if it's the same
            if append and pdf_copy.exists():
                existing_md5 = hashlib.md5(pdf_copy.read_bytes()).hexdigest()
                new_md5 = hashlib.md5(pdf_path.read_bytes()).hexdigest()
                if existing_md5 == new_md5:
                    print(f"[SKIP] {pdf_path.name} already exists with same content")
                    continue
            
            # Copy and extract
            meta, text = copy_and_extract(pdf_path, dest_dir)
            combined_text += f"\n\n--- From {pdf_path.name} ---\n\n{text}"
            
            # Add metadata
            pdf_meta = {
                "file": meta["file"],
                "md5": meta["md5"],
                "chars": meta["chars"],
                "added": datetime.utcnow().isoformat(),
                "source_path": str(pdf_path)
            }
            new_pdfs.append(pdf_meta)
            
            print(f"[OK] Processed {pdf_path.name} ({meta['chars']} chars)")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_path}: {e}")
    
    # Update spec
    existing_spec["pdfs"].extend(new_pdfs)
    existing_spec["total_chars"] = sum(pdf["chars"] for pdf in existing_spec["pdfs"])
    existing_spec["last_updated"] = datetime.utcnow().isoformat()
    
    # Save updated spec
    spec_file.write_text(json.dumps(existing_spec, indent=2))
    
    # Update seed.txt
    seed_file = dest_dir / "seed.txt"
    if append and seed_file.exists():
        # Append new text to existing seed
        existing_seed = seed_file.read_text()
        combined_seed = existing_seed + combined_text
    else:
        combined_seed = combined_text
    
    # Truncate to max chars
    max_chars = existing_spec.get("seed_max_chars", 50000)
    seed_file.write_text(combined_seed[:max_chars])
    
    return existing_spec

def add_pdfs_to_server(server_name: str, pdf_paths: List[Path]) -> bool:
    """
    Add PDFs to an existing server.
    
    Args:
        server_name: Name of the server to update
        pdf_paths: List of PDF paths to add
    
    Returns:
        True if successful, False otherwise
    """
    # Find the server directory
    base_dir = Path(__file__).parent.parent / "agents" / server_name
    
    if not base_dir.exists():
        print(f"[ERROR] Server '{server_name}' not found at {base_dir}")
        return False
    
    # Add PDFs
    spec = copy_and_extract_multiple(pdf_paths, base_dir, append=True)
    
    print(f"\n[SUMMARY] Server '{server_name}' now has {len(spec['pdfs'])} PDFs")
    print(f"Total characters: {spec['total_chars']}")
    
    return True

def list_server_pdfs(server_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    List all PDFs associated with a server.
    
    Args:
        server_name: Name of the server
    
    Returns:
        List of PDF metadata or None if server not found
    """
    base_dir = Path(__file__).parent.parent / "agents" / server_name
    spec_file = base_dir / "spec.json"
    
    if not spec_file.exists():
        return None
    
    spec = json.loads(spec_file.read_text())
    return spec.get("pdfs", [])
