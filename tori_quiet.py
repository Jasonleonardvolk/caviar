#!/usr/bin/env python3
"""
Quick startup optimization for TORI - suppress warnings and speed up initialization
"""

import os
import sys
import warnings

# Set environment variables before any imports
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['TORI_FAST_STARTUP'] = '1'
os.environ['TORI_DISABLE_MESH_CHECK'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings if used

# Configure warning filters
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*already exists.*")
warnings.filterwarnings("ignore", message=".*shadows an attribute.*")

# Monkey patch the warning system for MCP
original_warn = warnings.warn
def filtered_warn(message, category=None, stacklevel=1, source=None):
    # Skip specific warnings
    if isinstance(message, str):
        skip_patterns = [
            "already exists",
            "shadows an attribute",
            "Field name"
        ]
        if any(pattern in message for pattern in skip_patterns):
            return
    original_warn(message, category, stacklevel, source)

warnings.warn = filtered_warn

# Import logging and set levels
import logging

# Suppress noisy loggers
noisy_loggers = [
    "mcp.server.fastmcp.tools.tool_manager",
    "mcp.server.fastmcp.resources.resource_manager", 
    "mcp.server.fastmcp.prompts.manager",
    "ingest_pdf.pipeline.quality"
]

for logger_name in noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Create a custom logger for clean output
class CleanLogger:
    def __init__(self):
        self.start_time = None
        
    def info(self, message):
        # Only show important messages
        important_keywords = ["ready", "started", "initialized", "complete", "success"]
        if any(keyword in message.lower() for keyword in important_keywords):
            print(f"✅ {message}")
    
    def warning(self, message):
        # Suppress duplicate warnings
        if "already exists" not in message:
            print(f"⚠️  {message}")
    
    def error(self, message):
        print(f"❌ {message}")

# Patch the enhanced launcher
def launch_tori_clean():
    """Launch TORI with clean output"""
    print("🚀 TORI Clean Launch")
    print("=" * 60)
    
    # Redirect stderr to suppress warnings
    class FilteredStderr:
        def __init__(self):
            self.terminal = sys.stderr
            self.suppress_patterns = [
                "WARNING:",
                "UserWarning:",
                "already exists",
                "shadows an attribute",
                "0 concepts"
            ]
        
        def write(self, message):
            if not any(pattern in message for pattern in self.suppress_patterns):
                self.terminal.write(message)
        
        def flush(self):
            self.terminal.flush()
    
    # Apply stderr filter
    old_stderr = sys.stderr
    sys.stderr = FilteredStderr()
    
    try:
        # Import and run enhanced launcher
        import enhanced_launcher
        enhanced_launcher.main()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down gracefully...")
    finally:
        sys.stderr = old_stderr

if __name__ == "__main__":
    launch_tori_clean()
