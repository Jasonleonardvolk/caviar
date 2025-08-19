"""
TORI Login Issue Diagnostics
Checks why the login screen isn't appearing
"""

import requests
import json
import sys
from pathlib import Path

print("🔍 TORI Login System Diagnostics")
print("=" * 60)

# Check API endpoints
api_base = "http://localhost:8002"
frontend_base = "http://localhost:5173"

print("\n1️⃣ Checking API Authentication Endpoints...")
auth_endpoints = [
    "/api/auth/status",
    "/api/auth/login", 
    "/api/auth/user",
    "/api/auth/me",
    "/api/user/profile",
    "/login"
]

for endpoint in auth_endpoints:
    try:
        resp = requests.get(f"{api_base}{endpoint}", timeout=2)
        print(f"   {endpoint}: {resp.status_code} - {resp.reason}")
        if resp.status_code == 200:
            print(f"      Response: {resp.text[:100]}...")
    except Exception as e:
        print(f"   {endpoint}: ❌ {type(e).__name__}")

print("\n2️⃣ Checking Frontend Routes...")
frontend_routes = [
    "/",
    "/login",
    "/auth/login",
    "/signin"
]

for route in frontend_routes:
    try:
        resp = requests.get(f"{frontend_base}{route}", timeout=2, allow_redirects=False)
        print(f"   {route}: {resp.status_code}")
        if 'location' in resp.headers:
            print(f"      Redirects to: {resp.headers['location']}")
    except Exception as e:
        print(f"   {route}: ❌ {type(e).__name__}")

print("\n3️⃣ Checking CORS Configuration...")
try:
    # Test CORS preflight
    headers = {
        'Origin': 'http://localhost:5173',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'content-type'
    }
    resp = requests.options(f"{api_base}/api/auth/login", headers=headers, timeout=2)
    print(f"   CORS Preflight: {resp.status_code}")
    cors_headers = {k: v for k, v in resp.headers.items() if 'access-control' in k.lower()}
    for header, value in cors_headers.items():
        print(f"      {header}: {value}")
except Exception as e:
    print(f"   CORS Check: ❌ {type(e).__name__}")

print("\n4️⃣ Frontend Auth Configuration...")
# Check if there's a frontend config file
frontend_dir = Path("frontend")
if frontend_dir.exists():
    config_files = list(frontend_dir.glob("**/auth*.{js,ts,jsx,tsx,json}"))
    if config_files:
        print("   Found auth config files:")
        for f in config_files[:5]:  # Show first 5
            print(f"      {f.relative_to('.')}")
    else:
        print("   ⚠️ No auth config files found in frontend/")

print("\n" + "=" * 60)
print("💡 RECOMMENDATIONS:")

# Check if we found auth endpoints
if not any(endpoint in ["/api/auth/login", "/api/auth/status"] for endpoint in auth_endpoints):
    print("\n❌ Missing authentication endpoints in API!")
    print("   → Add /api/auth/login and /api/auth/status endpoints to your API")
    
print("\n✅ Quick Fix - Add this to your API (prajna_api.py or similar):")
print("""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

auth_router = APIRouter(prefix="/api/auth")

class LoginRequest(BaseModel):
    username: str
    password: str

@auth_router.post("/login")
async def login(credentials: LoginRequest):
    # Simple auth for testing
    if credentials.username == "admin" and credentials.password == "admin":
        return {"success": True, "token": "test-token-123", "user": {"username": "admin"}}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@auth_router.get("/status")
async def auth_status():
    return {"authenticated": False, "user": None}

# Don't forget to include this router in your main app:
# app.include_router(auth_router)
""")

print("\n🔧 Frontend Quick Fix:")
print("   If the login form component is missing, check:")
print("   - frontend/src/routes/login/+page.svelte (SvelteKit)")
print("   - frontend/src/components/Login.jsx (React)")
print("   - frontend/src/views/Login.vue (Vue)")
