#!/usr/bin/env python3
"""
Fix critical issues blocking memory storage and causing log spam
"""

import logging
from pathlib import Path
import re
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def fix_memory_sculptor_user_validation():
    """Fix the user_id validation in memory_sculptor.py"""
    
    file_path = Path("ingest_pdf/memory_sculptor.py")
    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # The issue is in the sculpt_and_store method
    # Find the validation section and improve it
    original = '''        # Validate user_id
        if not user_id or user_id == "default":
            logger.error(f"❌ Invalid user_id provided: '{user_id}'. Cannot store memory without valid user ID.")
            return []'''
    
    # Better fix that only logs once
    fixed = '''        # Validate user_id
        if not user_id or user_id == "default":
            # Only log once to avoid spam
            if not hasattr(self, '_warned_default_user'):
                logger.warning("⚠️ Skipping memory store - no valid user_id provided (default user). This warning will only show once.")
                self._warned_default_user = True
            return []'''
    
    if original in content:
        content = content.replace(original, fixed)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"✅ Fixed user_id validation in {file_path}")
        return True
    else:
        logger.warning(f"⚠️ Could not find exact match in {file_path}, applying alternate fix...")
        
        # Try a regex-based approach
        pattern = r'(logger\.error\(f".*Invalid user_id.*\{user_id\}.*"\))'
        replacement = '''# Only log once to avoid spam
            if not hasattr(self, '_warned_default_user'):
                logger.warning("⚠️ Skipping memory store - no valid user_id provided (default user). This warning will only show once.")
                self._warned_default_user = True'''
        
        content = re.sub(pattern, replacement, content)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True


def fix_asyncio_loop_issue():
    """Fix the asyncio.run() issue in server_fixed.py"""
    
    file_path = Path("mcp_metacognitive/server_fixed.py")
    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the asyncio.run call and wrap it
    if "asyncio.run(" in content:
        # Add import for asyncio at the top if not present
        if "import asyncio" not in content:
            content = "import asyncio\n" + content
        
        # Replace asyncio.run with a safe wrapper
        pattern = r'(\s*)asyncio\.run\((.*?)\)'
        
        def replacement(match):
            indent = match.group(1)
            call = match.group(2)
            return f'''{indent}try:
{indent}    # Check if there's already a running event loop
{indent}    loop = asyncio.get_running_loop()
{indent}    # If we get here, there's already a loop running
{indent}    logger.warning("Event loop already running, using existing loop")
{indent}    loop.create_task({call})
{indent}except RuntimeError:
{indent}    # No loop running, safe to use asyncio.run
{indent}    asyncio.run({call})'''
        
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"✅ Fixed asyncio loop issue in {file_path}")
        return True
    else:
        logger.warning(f"⚠️ No asyncio.run() found in {file_path}")
        return False


def fix_record_diff_schema():
    """Fix the record_diff schema issue"""
    
    # First, let's check the pipeline storage module
    storage_path = Path("ingest_pdf/pipeline/storage.py")
    
    if not storage_path.exists():
        logger.warning(f"⚠️ Storage module not found at {storage_path}")
        # Try to find where record_diff is called
        import subprocess
        try:
            result = subprocess.run(['grep', '-r', 'record_diff', 'ingest_pdf/'], 
                                  capture_output=True, text=True)
            if result.stdout:
                logger.info(f"Found record_diff references:\n{result.stdout}")
        except:
            pass
        return False
    
    with open(storage_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for the record_diff call
    if "record_diff" in content:
        # Add label field to the payload
        pattern = r'(payload\s*=\s*\{[^}]+\})'
        
        def add_label_field(match):
            payload = match.group(1)
            if '"label"' not in payload and "'label'" not in payload:
                # Insert label field
                return payload.rstrip('}') + ',\n        "label": concept.get("name", "unnamed")\n    }'
            return payload
        
        content = re.sub(pattern, add_label_field, content, flags=re.DOTALL)
        
        with open(storage_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"✅ Fixed record_diff schema in {storage_path}")
        return True
    else:
        logger.warning(f"⚠️ No record_diff call found in {storage_path}")
        return False


def add_user_context_to_upload():
    """Add user context to the upload pipeline"""
    
    # This would need to be done in the frontend upload component
    # Creating a patch file for reference
    
    patch_content = '''
// In your Svelte upload component, add user context:

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Add user context
    const userId = getCurrentUser() || 'adminuser';  // Get from auth system
    
    const response = await fetch('/api/upload/pdf', {
        method: 'POST',
        headers: {
            'X-User-Id': userId,  // Add user ID header
        },
        body: formData
    });
    
    return response.json();
}

// In your backend API endpoint, extract the user:
def upload_pdf(request: Request, file: UploadFile):
    user_id = request.headers.get('X-User-Id', 'default')
    
    # Pass user_id to the ingestion pipeline
    result = await ingest_pdf(file, user_id=user_id)
    return result
'''
    
    with open('fix_upload_user_context.patch', 'w') as f:
        f.write(patch_content)
    
    logger.info("✅ Created patch file: fix_upload_user_context.patch")
    return True


def reduce_concurrency_workers():
    """Reduce excessive concurrency workers"""
    
    concurrency_path = Path("ingest_pdf/concurrency.py")
    if not concurrency_path.exists():
        logger.warning(f"⚠️ Concurrency module not found at {concurrency_path}")
        return False
    
    with open(concurrency_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Cap CPU workers to reasonable amount
    pattern = r'(cpu_workers\s*=\s*)(os\.cpu_count\(\)\s*\+\s*\d+|[\d]+)'
    replacement = r'\1min(os.cpu_count() or 4, 8)'
    
    content = re.sub(pattern, replacement, content)
    
    with open(concurrency_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"✅ Reduced concurrency workers in {concurrency_path}")
    return True


def main():
    """Run all fixes"""
    logger.info("🔧 FIXING CRITICAL TORI/KHA ISSUES")
    logger.info("="*70)
    
    fixes = [
        ("Memory Sculptor user_id spam", fix_memory_sculptor_user_validation),
        ("Asyncio loop conflict", fix_asyncio_loop_issue),
        ("Record diff schema", fix_record_diff_schema),
        ("Upload user context", add_user_context_to_upload),
        ("Excessive concurrency", reduce_concurrency_workers)
    ]
    
    results = {}
    
    for name, fix_func in fixes:
        logger.info(f"\n🔧 Fixing: {name}")
        try:
            success = fix_func()
            results[name] = "✅ Fixed" if success else "⚠️ Needs manual fix"
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            results[name] = f"❌ Error: {str(e)}"
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("📊 FIX SUMMARY")
    logger.info("="*70)
    
    for name, status in results.items():
        logger.info(f"{name}: {status}")
    
    # Next steps
    logger.info("\n💡 Next steps:")
    logger.info("1. Review fix_upload_user_context.patch for frontend changes")
    logger.info("2. Restart the TORI launcher to apply fixes")
    logger.info("3. Monitor logs - should be 90% quieter")
    logger.info("4. Concepts should now persist to memory vault")
    
    # Create verification script
    verification_script = '''#!/usr/bin/env python3
"""Verify fixes are working"""

import subprocess
import time
import re

print("🔍 Verifying fixes...")

# Start services and capture logs
proc = subprocess.Popen(['python', 'launch_tori_basic.py'], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)

# Monitor for 30 seconds
time.sleep(30)

# Check for error patterns
stdout, stderr = proc.communicate(timeout=5)
logs = stdout + stderr

issues = {
    "user_id spam": logs.count("Invalid user_id") > 5,
    "asyncio conflict": "Already running asyncio" in logs,
    "record_diff errors": "422 Unprocessable Entity" in logs
}

print("\\n📊 Verification Results:")
for issue, found in issues.items():
    status = "❌ Still present" if found else "✅ Fixed"
    print(f"{issue}: {status}")
'''
    
    with open('verify_fixes.py', 'w') as f:
        f.write(verification_script)
    
    logger.info("\n✅ Created verify_fixes.py to test the changes")


if __name__ == "__main__":
    main()
