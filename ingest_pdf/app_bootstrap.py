"""
app_bootstrap.py

Example application bootstrap showing proper logging initialization.
This should be the entry point of your application.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.logging_config import ToriLoggerConfig, configure_preset


def initialize_logging():
    """Initialize logging configuration for the application."""
    
    # Determine environment
    environment = os.environ.get('ENVIRONMENT', 'development').lower()
    
    # Use preset based on environment
    if environment == 'production':
        configure_preset('production', 
                        log_file=os.environ.get('LOG_FILE', 'tori_ingest.log'))
    elif environment == 'testing':
        configure_preset('testing')
    else:
        # Development mode
        configure_preset('development')
    
    # Override with any environment-specific settings
    if os.environ.get('LOG_CONFIG_FILE'):
        import json
        with open(os.environ['LOG_CONFIG_FILE'], 'r') as f:
            config = json.load(f)
            ToriLoggerConfig.configure_from_dict(config)


def main():
    """Main application entry point."""
    
    # Initialize logging first
    initialize_logging()
    
    # Now import modules that use logging
    from pipeline.pipeline_improved import (
        ingest_pdf_async, 
        ingest_pdf_clean,
        preload_concept_database,
        get_logger
    )
    
    # Get logger for this module
    logger = get_logger(__name__)
    
    logger.info("Application starting...")
    
    # Preload resources
    logger.info("Preloading concept database...")
    preload_concept_database()
    
    # Example: Process a PDF
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            result = ingest_pdf_clean(pdf_path)
            logger.info(f"Successfully processed: {result['concept_count']} concepts extracted")
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}", exc_info=True)
    else:
        logger.info("No PDF specified. Usage: python app_bootstrap.py <pdf_path>")
    
    logger.info("Application completed")


if __name__ == "__main__":
    main()
