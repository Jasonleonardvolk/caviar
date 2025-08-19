    def wait_for_api_health(self, port, max_retries=30):
        """Wait for API health check to succeed before proceeding"""
        self.logger.info(f"⏳ Waiting for API to be healthy on port {port}...")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f'http://localhost:{port}/api/health', timeout=2)
                if response.status_code == 200:
                    health_data = response.json()
                    self.logger.info(f"✅ API is healthy: {health_data.get('status', 'unknown')}")
                    return True
            except requests.exceptions.RequestException:
                if attempt % 5 == 0 and attempt > 0:
                    self.logger.info(f"⏳ Still waiting for API... ({attempt}/{max_retries})")
            
            time.sleep(1)
        
        self.logger.error(f"❌ API failed to become healthy after {max_retries} seconds")
        return False
