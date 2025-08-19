# start_tori_services.py - Start all TORI services with management
import subprocess
import time
import sys
import os
import signal

class ToriServiceManager:
    def __init__(self):
        self.embedding_process = None
        
    def start_embedding_service(self):
        """Start the embedding service"""
        print("🚀 Starting embedding service...")
        self.embedding_process = subprocess.Popen(
            [sys.executable, "run_embedding_service_graceful.py"],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        
        # Wait for service to be ready
        print("⏳ Waiting for service to initialize...")
        time.sleep(5)
        
        # Check if service is running
        import requests
        try:
            response = requests.get("http://localhost:8080/health")
            if response.status_code == 200:
                print("✅ Embedding service is running on port 8080")
                return True
        except:
            print("❌ Embedding service failed to start")
            return False
    
    def stop_embedding_service(self):
        """Stop the embedding service gracefully"""
        if self.embedding_process:
            print("🛑 Stopping embedding service...")
            if sys.platform == "win32":
                # Send CTRL_BREAK_EVENT on Windows
                self.embedding_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Send SIGTERM on Unix
                self.embedding_process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.embedding_process.wait(timeout=10)
                print("✅ Embedding service stopped")
            except subprocess.TimeoutExpired:
                print("⚠️ Force killing embedding service...")
                self.embedding_process.kill()
                self.embedding_process.wait()
    
    def status(self):
        """Check service status"""
        import requests
        try:
            response = requests.get("http://localhost:8080/health")
            if response.status_code == 200:
                health = response.json()
                print("✅ Embedding service: RUNNING")
                print(f"   - Model: {health.get('model', 'unknown')}")
                print(f"   - Device: {health.get('device', 'unknown')}")
                print(f"   - Cache size: {health.get('cache_size', 0)}")
            else:
                print("❌ Embedding service: NOT RESPONDING")
        except:
            print("❌ Embedding service: NOT RUNNING")

def main():
    manager = ToriServiceManager()
    
    print("TORI Service Manager")
    print("===================")
    
    while True:
        print("\nOptions:")
        print("1. Start embedding service")
        print("2. Stop embedding service")
        print("3. Check status")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ")
        
        if choice == "1":
            manager.start_embedding_service()
        elif choice == "2":
            manager.stop_embedding_service()
        elif choice == "3":
            manager.status()
        elif choice == "4":
            print("Shutting down...")
            manager.stop_embedding_service()
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down services...")
        ToriServiceManager().stop_embedding_service()
