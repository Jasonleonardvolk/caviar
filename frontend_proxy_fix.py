"""
Frontend Proxy Fix - Add this to enhanced_launcher.py after line 1626 (before start_api_server)
"""

def wait_for_api_health(self, port, max_attempts=30):
    """Wait for API server to be healthy before continuing"""
    self.logger.info(f"⏳ Waiting for API health check on port {port}...")
    
    for i in range(max_attempts):
        try:
            response = requests.get(f'http://localhost:{port}/api/health', timeout=2)
            if response.status_code == 200:
                self.logger.info("✅ API server is healthy!")
                return True
        except requests.exceptions.RequestException:
            if i % 5 == 0 and i > 0:
                self.logger.info(f"⏳ Still waiting for API... ({i}/{max_attempts} attempts)")
            time.sleep(1)
    
    self.logger.warning("⚠️ API health check timeout - proceeding anyway")
    return False

# Then modify the launch() method around line 1660:
# Add this after starting the API server in a thread:

# Start API server in background thread
api_thread = threading.Thread(
    target=self.start_api_server,
    args=(self.api_port,),
    daemon=True,
    name="APIServer"
)
api_thread.start()

# Wait for API to be healthy
self.wait_for_api_health(self.api_port)

# NOW start frontend (it will connect successfully)
frontend_started = self.start_frontend_services_enhanced()
