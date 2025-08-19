"""
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
