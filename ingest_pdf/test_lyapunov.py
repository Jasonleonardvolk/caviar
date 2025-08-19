import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to allow imports when running from this directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ingest_pdf.pipeline import ingest_pdf_and_update_index
from ingest_pdf.lyapunov import concept_predictability, document_chaos_profile


def run_lyapunov_analysis(pdf_path, output_dir=None):
    """
    Run Lyapunov analysis on the given PDF and visualize results.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save visualizations
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(pdf_path), 'lyapunov_analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process PDF and run analysis
    json_output = os.path.join(output_dir, f"{Path(pdf_path).stem}_concepts.json")
    npz_output = os.path.join(output_dir, f"{Path(pdf_path).stem}_concepts.npz")
    
    print(f"Processing PDF: {pdf_path}")
    result = ingest_pdf_and_update_index(pdf_path, npz_output, json_out=json_output)
    print(f"Extracted {result['concept_count']} concepts")
    
    # Load the results for analysis
    with open(json_output, 'r') as f:
        data = json.load(f)
    
    # Separate concept data from chaos profile
    concepts = [item for item in data if 'type' not in item]
    chaos_profile = next((item for item in data if item.get('type') == 'document_chaos_profile'), None)
    
    # Display predictability scores
    print("\nConcept Predictability Analysis:")
    print("=" * 40)
    print(f"{'Concept Name':<30} {'Predictability':<15} {'Interpretation'}")
    print("-" * 80)
    
    for concept in sorted(concepts, key=lambda x: x['predictability_score'], reverse=True):
        name = concept['name']
        pred_score = concept['predictability_score']
        
        # Interpret the score
        if pred_score > 0.8:
            interpretation = "Highly predictable (formulaic)"
        elif pred_score > 0.6:
            interpretation = "Somewhat predictable"
        elif pred_score > 0.4:
            interpretation = "Neutral"
        elif pred_score > 0.2:
            interpretation = "Somewhat chaotic"
        else:
            interpretation = "Highly chaotic (creative/novel)"
            
        print(f"{name[:30]:<30} {pred_score:.3f}           {interpretation}")
    
    # Create visualizations
    if chaos_profile and 'values' in chaos_profile:
        plt.figure(figsize=(10, 6))
        plt.plot(chaos_profile['values'])
        plt.title('Document Chaos Profile')
        plt.xlabel('Document Position (Sliding Window)')
        plt.ylabel('Chaos Level (0=Predictable, 1=Chaotic)')
        plt.grid(True, alpha=0.3)
        
        # Highlight regions of interest
        chaos_values = np.array(chaos_profile['values'])
        chaotic_regions = np.where(chaos_values > 0.7)[0]
        predictable_regions = np.where(chaos_values < 0.3)[0]
        
        if len(chaotic_regions) > 0:
            plt.scatter(chaotic_regions, chaos_values[chaotic_regions], 
                        color='red', alpha=0.7, label='Chaotic/Creative')
        
        if len(predictable_regions) > 0:
            plt.scatter(predictable_regions, chaos_values[predictable_regions], 
                       color='green', alpha=0.7, label='Predictable/Formulaic')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{Path(pdf_path).stem}_chaos_profile.png"))
        print(f"\nChaos profile visualization saved to: {output_dir}")
    
    # Create concept predictability visualization
    plt.figure(figsize=(12, 6))
    concept_names = [c['name'][:20] for c in concepts]
    pred_scores = [c['predictability_score'] for c in concepts]
    
    # Sort by predictability
    sorted_indices = np.argsort(pred_scores)
    concept_names = [concept_names[i] for i in sorted_indices]
    pred_scores = [pred_scores[i] for i in sorted_indices]
    
    # Create color gradient
    colors = plt.cm.RdYlGn(np.array(pred_scores))
    
    plt.barh(concept_names, pred_scores, color=colors)
    plt.xlabel('Predictability Score (0=Chaotic, 1=Predictable)')
    plt.ylabel('Concept')
    plt.title('Concept Predictability Analysis')
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    plt.xlim(0, 1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{Path(pdf_path).stem}_concept_predictability.png"))
    
    print(f"Concept predictability visualization saved to: {output_dir}")
    print("\nAnalysis complete!")
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_lyapunov.py <path_to_pdf> [output_directory]")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_lyapunov_analysis(pdf_path, output_dir)
