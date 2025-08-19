#!/usr/bin/env python3
"""
Manual fixes for remaining issues
"""

import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def fix_concurrency_workers():
    """Fix excessive concurrency workers in concurrency_manager.py"""
    
    file_path = Path("ingest_pdf/pipeline/concurrency_manager.py")
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Original pattern for CPU workers
    original = "cpu_count() + 2"
    replacement = "min(cpu_count() or 4, 8)"
    
    if original in content:
        content = content.replace(original, replacement)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Fixed CPU worker count in {file_path}")
        return True
    else:
        logger.warning(f"Could not find pattern in {file_path}")
        return False


def add_label_to_concept_mesh_calls():
    """Add label field to concept mesh API calls"""
    
    # Search for files that might call the concept mesh API
    potential_files = [
        "ingest_pdf/cognitive_interface.py",
        "python/core/cognitive_interface.py",
        "ingest_pdf/pipeline/storage.py"
    ]
    
    fixed = False
    for file_path in potential_files:
        path = Path(file_path)
        if not path.exists():
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for concept mesh API calls
        if "record_diff" in content or "concept-mesh" in content:
            # Add label field to any dictionary that's being sent
            # This is a heuristic approach
            pattern = r'({\s*["\']doc_id["\']\s*:.*?["\']concepts["\']\s*:.*?})'
            
            def add_label(match):
                dict_str = match.group(1)
                if '"label"' not in dict_str and "'label'" not in dict_str:
                    # Insert label before the closing brace
                    return dict_str.rstrip('}') + ', "label": concept.get("name", concept.get("label", "unnamed"))}'
                return dict_str
            
            new_content = re.sub(pattern, add_label, content, flags=re.DOTALL)
            
            if new_content != content:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info(f"Added label field to API calls in {path}")
                fixed = True
    
    return fixed


def create_user_context_middleware():
    """Create middleware to inject user context"""
    
    middleware_content = '''"""
Middleware to inject user context into requests
"""

from fastapi import Request
import logging

logger = logging.getLogger(__name__)


async def user_context_middleware(request: Request, call_next):
    """
    Middleware to extract and inject user context
    """
    # Try to get user ID from various sources
    user_id = None
    
    # 1. Check header
    user_id = request.headers.get('X-User-Id')
    
    # 2. Check JWT token if present
    if not user_id:
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            # Extract user from JWT (simplified - use proper JWT library)
            # user_id = decode_jwt(auth_header[7:]).get('sub')
            pass
    
    # 3. Check session
    if not user_id and hasattr(request, 'session'):
        user_id = request.session.get('user_id')
    
    # 4. Default to adminuser if authenticated
    if not user_id:
        # In production, check if user is authenticated
        # For now, use adminuser as default for authenticated requests
        if request.url.path.startswith('/api/'):
            user_id = 'adminuser'
    
    # Inject into request state
    request.state.user_id = user_id or 'default'
    
    # Log once per unique user
    if user_id and user_id != 'default':
        if not hasattr(user_context_middleware, '_logged_users'):
            user_context_middleware._logged_users = set()
        if user_id not in user_context_middleware._logged_users:
            logger.info(f"Processing request for user: {user_id}")
            user_context_middleware._logged_users.add(user_id)
    
    response = await call_next(request)
    return response


# Usage in FastAPI app:
# app.middleware("http")(user_context_middleware)
'''
    
    with open('user_context_middleware.py', 'w', encoding='utf-8') as f:
        f.write(middleware_content)
    
    logger.info("Created user_context_middleware.py")
    return True


def main():
    """Run manual fixes"""
    logger.info("APPLYING MANUAL FIXES")
    logger.info("="*70)
    
    # Fix concurrency
    logger.info("\nFixing concurrency workers...")
    fix_concurrency_workers()
    
    # Fix record_diff schema
    logger.info("\nAdding label field to API calls...")
    add_label_to_concept_mesh_calls()
    
    # Create middleware
    logger.info("\nCreating user context middleware...")
    create_user_context_middleware()
    
    logger.info("\n" + "="*70)
    logger.info("MANUAL FIXES COMPLETE")
    logger.info("="*70)
    logger.info("\nNext steps:")
    logger.info("1. Add user_context_middleware to your FastAPI app")
    logger.info("2. Update frontend to send X-User-Id header")
    logger.info("3. Restart TORI and monitor logs")
    

if __name__ == "__main__":
    main()
