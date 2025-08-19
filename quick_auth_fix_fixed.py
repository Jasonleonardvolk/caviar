"""
Quick Auth Fix for TORI API
Adds missing authentication endpoints to get login working
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("[FIX] Applying Quick Auth Fix to TORI API...")
print("=" * 60)

# Find the main API file
api_files = [
    "api/main.py",
    "prajna_api.py", 
    "api.py",
    "main.py",
    "prajna/api/prajna_api.py"
]

api_file = None
for f in api_files:
    if Path(f).exists():
        api_file = Path(f)
        break

if not api_file:
    print("[ERROR] Could not find main API file!")
    print("   Please specify the path to your FastAPI main file")
    sys.exit(1)

print(f"[OK] Found API file: {api_file}")

# Read current content
content = api_file.read_text(encoding='utf-8')

# Check if auth endpoints already exist
if "/api/auth" in content or "auth_router" in content:
    print("[WARNING] Auth routes may already exist. Skipping auto-fix.")
    print("   Please check your API manually.")
    sys.exit(0)

# Create auth router code
auth_code = '''
# === TORI AUTH ENDPOINTS (QUICK FIX) ===
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional

auth_router = APIRouter(prefix="/api/auth", tags=["authentication"])

class LoginRequest(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    username: str
    email: Optional[str] = None
    role: str = "user"

@auth_router.post("/login")
async def login(credentials: LoginRequest, response: Response):
    """Simple login endpoint for testing"""
    # TODO: Replace with real authentication
    if credentials.username in ["admin", "user", "test"] and credentials.password == credentials.username:
        # Set a simple cookie for now
        response.set_cookie(
            key="tori_session",
            value=f"session_{credentials.username}_123",
            httponly=True,
            samesite="lax"
        )
        return {
            "success": True,
            "user": UserResponse(username=credentials.username, email=f"{credentials.username}@tori.local"),
            "message": "Login successful"
        }
    raise HTTPException(status_code=401, detail="Invalid username or password")

@auth_router.post("/logout")
async def logout(response: Response):
    """Logout endpoint"""
    response.delete_cookie("tori_session")
    return {"success": True, "message": "Logged out successfully"}

@auth_router.get("/status")
async def auth_status():
    """Check authentication status"""
    # TODO: Check real session
    return {
        "authenticated": False,
        "user": None
    }

@auth_router.get("/me")
async def get_current_user():
    """Get current user info"""
    # TODO: Get from real session
    return {
        "authenticated": False,
        "user": None
    }

# === END AUTH ENDPOINTS ===

'''

# Find where to insert the auth code (before app creation or after imports)
import_end = content.rfind("from")
if import_end == -1:
    import_end = content.rfind("import")

if import_end != -1:
    # Find the end of the import block
    import_section_end = content.find("\n\n", import_end)
    if import_section_end == -1:
        import_section_end = len(content)
    
    # Insert auth code after imports
    new_content = (
        content[:import_section_end] + 
        "\n\n" + auth_code + 
        content[import_section_end:]
    )
else:
    # Just prepend if we can't find imports
    new_content = auth_code + "\n\n" + content

# Find where to register the router
app_creation = new_content.find("app = FastAPI")
if app_creation != -1:
    # Find the next good place to add the router
    next_line = new_content.find("\n", app_creation)
    router_registration = "\n\n# Register auth router\napp.include_router(auth_router)\n"
    
    # Check if CORS is configured
    if "CORSMiddleware" not in new_content:
        cors_code = '''
# Configure CORS for frontend
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
'''
        router_registration = "\n" + cors_code + router_registration
    
    # Insert router registration
    insert_point = new_content.find("\n\n", next_line)
    if insert_point == -1:
        insert_point = next_line
    
    new_content = (
        new_content[:insert_point] + 
        router_registration + 
        new_content[insert_point:]
    )

# Backup original file
backup_file = api_file.with_suffix(f".backup_{int(Path(api_file).stat().st_mtime)}")
api_file.rename(backup_file)
print(f"[BACKUP] Original saved to: {backup_file}")

# Write new content
api_file.write_text(new_content, encoding='utf-8')
print(f"[OK] Auth endpoints added to: {api_file}")

print("\n[COMPLETE] AUTH FIX APPLIED!")
print("\nYou can now login with these test credentials:")
print("  Username: admin   Password: admin")
print("  Username: user    Password: user")
print("  Username: test    Password: test")

print("\n[WARNING] This is a temporary fix for development!")
print("   Replace with proper authentication before production.")

print("\n[NEXT STEPS]")
print("1. Restart your API server")
print("2. Clear browser cache/cookies")
print("3. Try accessing http://localhost:5173/login again")
