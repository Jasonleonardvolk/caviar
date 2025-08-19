# Add graceful shutdown functionality to main.py
# Insert this code after the CORS middleware setup and before @app.get("/")

# Graceful shutdown event handlers
@app.on_event("startup")
async def startup_event():
    """API startup event"""
    logger.info("🚀 TORI Universal Concept Extraction API starting up...")
    logger.info("📍 Available endpoints: /health, /extract, /extract/text, /docs")

@app.on_event("shutdown")
async def shutdown_event():
    """API shutdown event"""
    logger.info("🛑 TORI Universal Concept Extraction API shutting down...")
    logger.info("👋 Goodbye!")

# Optional: Remote shutdown endpoint (useful for development)
@app.get("/shutdown")
async def shutdown_api():
    """
    🛑 GRACEFUL SHUTDOWN ENDPOINT
    
    Allows remote shutdown of the API server.
    Useful for development and automated deployments.
    """
    import os
    import signal
    import asyncio
    
    logger.info("🛑 Shutdown requested via /shutdown endpoint")
    
    # Schedule shutdown after brief delay to allow response
    async def delayed_shutdown():
        await asyncio.sleep(1)  # Give time for response to be sent
        logger.info("🧨 Initiating graceful shutdown...")
        os.kill(os.getpid(), signal.SIGTERM)
    
    # Start shutdown in background
    asyncio.create_task(delayed_shutdown())
    
    return {
        "status": "shutdown_initiated",
        "message": "API server shutting down gracefully...",
        "timestamp": time.time()
    }
