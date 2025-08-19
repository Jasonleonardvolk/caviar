"""
Prajna Universal Pipeline Launcher
Runs the complete FastAPI backend on port 3000
"""
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Prajna Universal Pipeline on port 3000...")
    print("📄 PDF Upload: http://localhost:3000/api/upload")
    print("🧠 Extract Only: http://localhost:3000/api/extract") 
    print("📊 API Docs: http://localhost:3000/docs")
    print("❤️ Health: http://localhost:3000/api/health")
    print("-" * 60)
    
    uvicorn.run(
        "phase3_production_secure_dashboard_complete:app", 
        host="0.0.0.0", 
        port=3000, 
        log_level="info",
        reload=False  # Disable reload to avoid path issues
    )
