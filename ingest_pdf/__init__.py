"""
IngestPDF module - PDF processing and concept extraction
"""

# Simple load_concept_mesh function for the API
def load_concept_mesh():
    """Load concept mesh data from the latest diffs"""
    import json
    from pathlib import Path
    
    # First try the concept_mesh/concepts.json file
    mesh_file = Path(__file__).parent.parent / "concept_mesh" / "concepts.json"
    
    if mesh_file.exists():
        try:
            with open(mesh_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Return the data in the format the API expects
                if isinstance(data, dict) and 'concepts' in data:
                    return data['concepts']
                return data
        except Exception as e:
            print(f"Error loading concept mesh: {e}")
    
    # Alternative: look for concept diff files
    diffs_dir = Path(__file__).parent.parent / "data" / "concept_diffs"
    if diffs_dir.exists():
        all_diffs = []
        for diff_file in sorted(diffs_dir.glob("*.json")):
            try:
                with open(diff_file, 'r', encoding='utf-8') as f:
                    all_diffs.append(json.load(f))
            except:
                pass
        if all_diffs:
            return all_diffs
    
    # Return empty list if nothing found
    return []

# Import the main pipeline
try:
    from .pipeline import ingest_pdf_clean, ingest_pdf_async
except ImportError:
    try:
        from pipeline import ingest_pdf_clean, ingest_pdf_async
    except ImportError:
        ingest_pdf_clean = None
        ingest_pdf_async = None

__all__ = ['load_concept_mesh', 'ingest_pdf_clean', 'ingest_pdf_async']
