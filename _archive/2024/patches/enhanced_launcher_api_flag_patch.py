#!/usr/bin/env python3
"""
PATCH for enhanced_launcher.py - Add CLI flag to choose API mode

This patch adds a --api flag to choose between "quick" and "full" API at launch time.

INSTRUCTIONS:
1. Add this import at the top of enhanced_launcher.py (after other imports):
   import argparse

2. Add this argument parsing section BEFORE the main class definition:
"""

# ========== ADD THIS SECTION TO enhanced_launcher.py ==========

# Parse command line arguments early
def parse_launcher_args():
    parser = argparse.ArgumentParser(description="TORI Enhanced Launcher with API selection")
    parser.add_argument("--api", 
                       choices=["quick", "full"], 
                       default="full",
                       help="Which API to launch: 'quick' (minimal) or 'full' (all endpoints). Default: full")
    parser.add_argument("--api-port", 
                       type=int, 
                       default=None,
                       help="Override API port (default: 8002 for quick, 8001 for full)")
    parser.add_argument("--no-color", 
                       action="store_true",
                       help="Disable colored output")
    parser.add_argument("--skip-health-checks", 
                       action="store_true",
                       help="Skip health checks during startup")
    return parser.parse_args()

# Call this at the very beginning of your script
launcher_args = parse_launcher_args()

# ========== MODIFY THE API SERVER LAUNCH SECTION ==========

"""
Find the section in enhanced_launcher.py that looks like:

    async def _launch_api_server(self, host, port):
        ...
        from quick_api_server import app as quick_app
        ...
        uvicorn.run(quick_app, ...)

Replace it with:
"""

    async def _launch_api_server(self, host, port):
        """Launch API server with selected mode (quick or full)"""
        self.log_phase("API Server", "STARTING")
        
        # Dynamic API selection based on CLI flag
        if launcher_args.api == "quick":
            self.logger.info("🚀 Loading QUICK API (minimal endpoints)")
            from quick_api_server import app as api_app
            api_mode = "QUICK"
            default_port = 8002
        else:
            self.logger.info("🚀 Loading FULL API (all concept mesh endpoints)")
            try:
                from api import app as api_app
            except ImportError:
                self.logger.warning("Could not import full API from 'api' package, falling back to quick API")
                from quick_api_server import app as api_app
                api_mode = "QUICK (fallback)"
                default_port = 8002
            else:
                api_mode = "FULL"
                default_port = 8001
        
        # Use CLI port override if provided, otherwise use mode-specific default
        if launcher_args.api_port:
            port = launcher_args.api_port
            self.logger.info(f"Using custom API port from CLI: {port}")
        elif port is None:
            port = default_port
        
        self.logger.info(f"📍 Starting {api_mode} API on port {port}")
        self.logger.info(f"📚 API Documentation will be at: http://localhost:{port}/docs")
        
        # Add CORS if not already configured
        from fastapi.middleware.cors import CORSMiddleware
        if not any(isinstance(m, CORSMiddleware) for m in api_app.middleware):
            api_app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Configure server
        config = uvicorn.Config(
            api_app,
            host=host,
            port=port,
            reload=False,
            workers=1,
            log_level="info",
            access_log=True,
            loop="asyncio"
        )
        
        server = uvicorn.Server(config)
        
        # Store for cleanup
        self.servers['api'] = server
        self.services_info['api'] = {
            'name': f'{api_mode} API Server',
            'url': f"http://localhost:{port}",
            'docs': f"http://localhost:{port}/docs",
            'mode': api_mode.lower()
        }
        
        # Update port tracking
        self.api_port = port
        
        try:
            await server.serve()
        except Exception as e:
            self.logger.error(f"API server error: {e}")
            raise

# ========== UPDATE THE MAIN SECTION ==========

"""
At the bottom of enhanced_launcher.py, update the if __name__ == "__main__": section
to use the parsed arguments:
"""

if __name__ == "__main__":
    # Arguments already parsed at top of file
    if launcher_args.no_color:
        # Disable color output
        import os
        os.environ['NO_COLOR'] = '1'
    
    launcher = TORILauncher()
    
    # Show which API mode is selected
    print(f"\n🎯 Launching with {launcher_args.api.upper()} API mode")
    if launcher_args.api_port:
        print(f"📍 Using custom API port: {launcher_args.api_port}")
    
    try:
        asyncio.run(launcher.launch())
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutdown requested by user")
        asyncio.run(launcher.shutdown())
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
