#!/usr/bin/env python3
"""
Enhanced launcher fix - Start API first and wait for it before frontend
Apply this to enhanced_launcher.py around line 1660 in the launch() method
"""

# Replace the current launch sequence with this:

def launch(self):
    """Fixed launch sequence - API first, then frontend"""
    try:
        self.print_banner()
        
        # Step 1: Find and secure API port
        self.update_status("startup", "port_search", {"message": "Finding available port"})
        port = self.find_available_port(service_name="API")
        secured_port = self.secure_port_aggressively(port, "API")
        self.api_port = secured_port
        
        # Step 2: Start API server IN BACKGROUND THREAD FIRST
        self.logger.info("🚀 Starting API server in background thread...")
        api_thread = threading.Thread(
            target=self._run_api_server,
            args=(self.api_port,),
            daemon=True,
            name="APIServer"
        )
        api_thread.start()
        
        # Step 3: WAIT for API to be healthy before proceeding
        self.logger.info("⏳ Waiting for API server to be healthy...")
        api_ready = self._wait_for_api_health(self.api_port, max_attempts=60)
        
        if not api_ready:
            self.logger.error("❌ API server failed to start!")
            raise Exception("API server not responding")
        
        self.logger.info("✅ API server is healthy and ready!")
        
        # Step 4: NOW start other components (MCP, etc)
        self.logger.info("\n" + "=" * 50)
        self.logger.info("🧠 STARTING MCP METACOGNITIVE SERVER...")
        self.logger.info("=" * 50)
        mcp_started = self.start_mcp_metacognitive_server()
        
        # Step 5: Configure Prajna
        self.logger.info("\n" + "=" * 50)
        self.logger.info("🧠 CONFIGURING PRAJNA...")
        self.logger.info("=" * 50)
        prajna_configured = self.configure_prajna_integration_enhanced()
        
        # Step 6: Start core components
        self.start_core_python_components()
        self.start_stability_components()
        
        # Step 7: FINALLY start frontend (API is already up!)
        self.logger.info("\n" + "=" * 50)
        self.logger.info("🎨 STARTING FRONTEND...")
        self.logger.info("=" * 50)
        frontend_started = self.start_frontend_services_enhanced()
        
        # Save config and print status
        self.save_port_config(self.api_port, self.prajna_port, self.frontend_port, self.mcp_metacognitive_port)
        self.print_complete_system_ready(self.api_port, prajna_configured, frontend_started, mcp_started)
        
        # Keep main thread alive
        api_thread.join()
        
    except KeyboardInterrupt:
        self.logger.info("\n👋 Shutdown requested by user")
    except Exception as e:
        self.logger.error(f"❌ Launch failed: {e}")
        raise

def _wait_for_api_health(self, port, max_attempts=60):
    """Wait for API to respond to health checks"""
    import time
    import requests
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f'http://localhost:{port}/api/health', timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        
        if attempt % 5 == 0 and attempt > 0:
            self.logger.info(f"⏳ Still waiting for API... ({attempt}/{max_attempts})")
        time.sleep(0.25)  # 250ms as suggested
    
    return False

def _run_api_server(self, port):
    """Run API server in thread"""
    self.update_status("api_startup", "starting", {"port": port})
    self.start_api_server(port)
