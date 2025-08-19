#!/usr/bin/env python3
"""
Script to reset admin password in TORI application
Save this file and run it to reset your admin password
"""

import hashlib
import json
import os
from pathlib import Path

def hash_password(password):
    """Simple password hashing - you may need to adjust based on your auth system"""
    return hashlib.sha256(password.encode()).hexdigest()

def reset_admin_password():
    """Reset admin password"""
    print("TORI Admin Password Reset")
    print("=" * 40)
    
    # Get new password
    new_password = input("Enter new password for admin: ")
    confirm_password = input("Confirm new password: ")
    
    if new_password != confirm_password:
        print("❌ Passwords don't match!")
        return
    
    # Hash the password
    hashed_password = hash_password(new_password)
    
    # Create or update admin credentials
    admin_credentials = {
        "username": "admin",
        "password_hash": hashed_password,
        "role": "admin",
        "created_at": "2025-05-01",  # Since you mentioned May
        "reset_at": "2025-01-21"
    }
    
    # Save to a secure location
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    auth_file = config_dir / "admin_auth.json"
    with open(auth_file, 'w') as f:
        json.dump(admin_credentials, f, indent=2)
    
    print(f"✅ Admin password reset successfully!")
    print(f"📁 Credentials saved to: {auth_file}")
    print(f"🔐 Username: admin")
    print(f"🔑 Password: {new_password}")
    print("\n⚠️  Remember to:")
    print("1. Update your application to use this auth file")
    print("2. Secure the auth file with proper permissions")
    print("3. Consider using environment variables for production")
    
    # Create example auth integration code
    example_code = '''
# Example: How to integrate this in your FastAPI app

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import json
import hashlib

security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    with open("config/admin_auth.json") as f:
        admin_data = json.load(f)
    
    password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
    
    if credentials.username == admin_data["username"] and password_hash == admin_data["password_hash"]:
        return credentials.username
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Basic"},
    )

# Use in your routes:
# @app.get("/admin", dependencies=[Depends(verify_admin)])
'''
    
    example_file = "admin_auth_example.py"
    with open(example_file, 'w') as f:
        f.write(example_code)
    
    print(f"\n📄 Example integration code saved to: {example_file}")

if __name__ == "__main__":
    reset_admin_password()
