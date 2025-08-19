# process_pdf_through_tori.py - Process PDFs through the TORI API
import requests
import json
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def process_pdf_through_tori(pdf_path: str, api_port: int = 8002):
    """Process PDF through TORI's API endpoints"""
    
    print(f"📚 Processing PDF through TORI API")
    print(f"📄 File: {Path(pdf_path).name}")
    
    # Check API health first
    try:
        health_response = requests.get(f"http://localhost:{api_port}/api/health")
        if health_response.status_code == 200:
            print(f"✅ API is healthy on port {api_port}")
        else:
            print(f"❌ API health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print(f"💡 Make sure enhanced_launcher.py is running")
        return
    
    # Upload and process the PDF
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
            
            print(f"📤 Uploading PDF...")
            start_time = time.time()
            
            # Try the upload endpoint
            response = requests.post(
                f"http://localhost:{api_port}/api/upload",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                elapsed = time.time() - start_time
                
                print(f"\n✅ SUCCESS!")
                print(f"⏱️  Processing time: {elapsed:.1f}s")
                print(f"📊 Results:")
                print(f"  - Concepts extracted: {result.get('concept_count', 0)}")
                print(f"  - Status: {result.get('status', 'unknown')}")
                
                # Show top concepts if available
                if 'concepts' in result and result['concepts']:
                    print(f"\n🏆 Top 5 Concepts:")
                    for i, concept in enumerate(result['concepts'][:5], 1):
                        name = concept.get('name', 'Unknown')
                        score = concept.get('score', 0)
                        print(f"  {i}. {name} (score: {score:.3f})")
                
                return result
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")

if __name__ == "__main__":
    # Your PDF path
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    
    # Check if API port config exists
    try:
        with open("api_port.json", "r") as f:
            config = json.load(f)
            api_port = config.get("api_port", 8002)
    except:
        api_port = 8002
    
    process_pdf_through_tori(pdf_path, api_port)
