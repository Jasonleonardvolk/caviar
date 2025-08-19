from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# process_soliton_pdf.py - Process the Living Soliton Memory Systems PDF
import asyncio
from core.canonical_ingestion_production_fixed import ingest_file_production

async def main():
    pdf_path = r'{PROJECT_ROOT}\docs\Living Soliton Memory Systems.pdf'
    print('🚀 Processing Living Soliton Memory Systems PDF...')
    
    try:
        result = await ingest_file_production(pdf_path)
        
        if result.success:
            print(f'\n✅ Ingestion successful!')
            print(f'  - Concepts extracted: {result.concepts_extracted}')
            print(f'  - Embedding dimensions: {result.embedding_result.get("dimensions", "unknown")}')
            print(f'  - Processing time: {result.processing_time:.2f}s')
            
            if result.penrose_verification:
                print(f'\n🔍 Penrose Verification:')
                print(f'  - Status: {result.penrose_verification.status}')
                print(f'  - Pass rate: {result.penrose_verification.metadata.get("pass_rate", 0):.3f}')
                print(f'  - Geometric score: {result.penrose_verification.geometric_score:.3f}')
                print(f'  - Phase coherence: {result.penrose_verification.phase_coherence:.3f}')
                print(f'  - Semantic stability: {result.penrose_verification.semantic_stability:.3f}')
            
            print(f'\n📊 Quality Metrics:')
            for metric, value in result.quality_metrics.items():
                if isinstance(value, float):
                    print(f'  - {metric}: {value:.3f}')
            
            print(f'\n📁 Archive ID: {result.archive_id}')
            print(f'\n💡 This PDF about soliton memory systems has been successfully indexed!')
            
        else:
            print(f'\n❌ Ingestion failed')
            print(f'  - Error archived at: {result.archive_id}')
            # Try to read error details
            import json
            error_path = f'./psi_archive/{result.archive_id}.json'
            try:
                with open(error_path, 'r') as f:
                    error_data = json.load(f)
                    print(f'  - Error details: {error_data.get("error", "Unknown error")}')
            except:
                pass
                
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
