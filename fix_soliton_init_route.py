"""
Add missing /init route to soliton_production.py
This creates an alias for the existing /initialize route
"""

# Add this patch file to fix the soliton routes
import sys
from pathlib import Path

# Read the soliton_production.py file
soliton_file = Path(__file__).parent / "api" / "routes" / "soliton_production.py"
content = soliton_file.read_text()

# Find where to insert the new route (after the /initialize route)
insert_pos = content.find('@router.post("/initialize"')
if insert_pos == -1:
    print("ERROR: Could not find /initialize route")
    sys.exit(1)

# Find the end of the initialize function
end_pos = content.find('\n@router.post("/store")', insert_pos)
if end_pos == -1:
    print("ERROR: Could not find /store route")
    sys.exit(1)

# Create the /init route that calls initialize
init_route = '''
@router.post("/init", response_model=SolitonInitResponse)
async def init_soliton(background_tasks: BackgroundTasks):
    """Initialize Soliton memory (alias for /initialize for compatibility)"""
    # Default to 'adminuser' if not provided
    request = SolitonInitRequest(user_id="adminuser", lattice_reset=False)
    return await initialize_soliton(request, background_tasks)
'''

# Insert the new route
new_content = content[:end_pos] + init_route + content[end_pos:]

# Write back
soliton_file.write_text(new_content)
print("✅ Added /init route to soliton_production.py")
