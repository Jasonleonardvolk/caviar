#!/usr/bin/env python3
"""
Start Prajna Production System
==============================

This script launches the complete Prajna system with all components:
- Prajna API server (FastAPI)
- WebSocket streaming support
- Alien overlay audit system
- Ghost feedback analysis
- Memory system integration (Soliton + Concept Mesh)

Usage:
    python start_prajna.py [options]
    
Options:
    --config CONFIG_FILE    Load configuration from file
    --port PORT            API server port (default: 8001)
    --host HOST            API server host (default: 0.0.0.0)
    --debug                Enable debug mode
    --demo                 Enable demo mode (no real model loading)
    --log-level LEVEL      Set log level (DEBUG, INFO, WARNING, ERROR)
    --no-websocket         Disable WebSocket streaming
"""

import argparse
import asyncio
import logging
import signal
import sys
import uvicorn
from pathlib import Path

# Add prajna to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prajna.config.prajna_config import load_config, PrajnaConfig
from prajna.api.prajna_api import app

logger = logging.getLogger("prajna.launcher")

class PrajnaLauncher:
    """Main launcher for Prajna production system"""
    
    def __init__(self, config: PrajnaConfig):
        self.config = config
        self.server = None
        self.shutdown_event = asyncio.Event()
        
    async def start(self):
        """Start Prajna system"""
        try:
            logger.info("🚀 Starting Prajna - TORI's Voice and Language Model")
            logger.info(f"📋 Configuration: {self.config.config_source}")
            logger.info(f"🔧 Model: {self.config.model_type} on {self.config.device}")
            logger.info(f"🌐 API: {self.config.api_host}:{self.config.api_port}")
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Configure uvicorn
            uvicorn_config = uvicorn.Config(
                app=app,
                host=self.config.api_host,
                port=self.config.api_port,
                log_level=self.config.log_level.lower(),
                access_log=self.config.enable_performance_logging,
                reload=False,  # Disable in production
                workers=1,     # Single worker for now (model loading)
                timeout_keep_alive=self.config.api_timeout,
                limit_concurrency=self.config.max_concurrent_requests,
            )
            
            # Create and start server
            self.server = uvicorn.Server(uvicorn_config)
            
            logger.info("✅ Prajna is ready to serve!")
            logger.info(f"🔗 API endpoint: http://{self.config.api_host}:{self.config.api_port}/api/answer")
            logger.info(f"🔗 Health check: http://{self.config.api_host}:{self.config.api_port}/api/health")
            logger.info(f"🔗 Documentation: http://{self.config.api_host}:{self.config.api_port}/docs")
            
            if self.config.enable_streaming:
                logger.info(f"⚡ WebSocket streaming: ws://{self.config.api_host}:{self.config.api_port}/api/stream")
            
            # Start the server
            await self.server.serve()
            
        except Exception as e:
            logger.error(f"❌ Failed to start Prajna: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown of Prajna system"""
        try:
            logger.info("🔄 Shutting down Prajna...")
            
            if self.server:
                self.server.should_exit = True
                await self.server.shutdown()
            
            logger.info("✅ Prajna shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Start Prajna - TORI's Voice and Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start_prajna.py                          # Start with default config
    python start_prajna.py --demo                   # Start in demo mode
    python start_prajna.py --config custom.json    # Use custom config
    python start_prajna.py --port 8080 --debug     # Debug mode on port 8080
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path (JSON format)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="API server port (default: 8001)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        help="API server host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Enable demo mode (no real model loading)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set log level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-websocket",
        action="store_true",
        help="Disable WebSocket streaming"
    )
    
    parser.add_argument(
        "--model-type",
        choices=["rwkv", "llama", "gpt", "custom", "demo"],
        help="Override model type"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Override device selection"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Build configuration overrides from command line
    overrides = {}
    
    if args.port:
        overrides["api_port"] = args.port
    
    if args.host:
        overrides["api_host"] = args.host
    
    if args.debug:
        overrides["debug_mode"] = True
        overrides["log_level"] = "DEBUG"
    
    if args.demo:
        overrides["enable_demo_mode"] = True
        overrides["model_type"] = "demo"
    
    if args.log_level:
        overrides["log_level"] = args.log_level
    
    if args.no_websocket:
        overrides["enable_streaming"] = False
    
    if args.model_type:
        overrides["model_type"] = args.model_type
    
    if args.device:
        overrides["device"] = args.device
    
    try:
        # Load configuration
        config = load_config(
            config_file=args.config,
            use_env=True,
            **overrides
        )
        
        # Create and start launcher
        launcher = PrajnaLauncher(config)
        
        # Run the system
        asyncio.run(launcher.start())
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
