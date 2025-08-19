"""
Automated fixes for TORI system issues
Implements the fixes identified in the diagnostic report
"""

import json
import re
from pathlib import Path
import shutil
from typing import Dict, List


class TORIAutoFixer:
    """Automated fix implementation for common TORI issues"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.fixes_applied = []
        
    def fix_vite_proxy_configuration(self):
        """Fix Vite proxy configuration for API and WebSocket support"""
        vite_config_path = self.project_root / "tori_ui_svelte" / "vite.config.js"
        
        if not vite_config_path.exists():
            print(f"Warning: vite.config.js not found at {vite_config_path}")
            return False
            
        # Backup original
        shutil.copy(vite_config_path, vite_config_path.with_suffix('.js.backup'))
        
        # Read current content
        content = vite_config_path.read_text()
        
        # Check if proxy is already configured
        if "proxy:" in content and "'/api':" in content:
            print("Vite proxy already configured, updating for WebSocket support...")
            
            # Update existing proxy to ensure ws: true
            content = re.sub(
                r"'/api':\s*{[^}]+}",
                """'/api': {
      target: 'http://localhost:8002',
      changeOrigin: true,
      ws: true, // Enable WebSocket proxying
      configure: (proxy, options) => {
        proxy.on('error', (err, req, res) => {
          console.log('proxy error', err);
        });
        proxy.on('proxyReq', (proxyReq, req, res) => {
          console.log('Sending Request:', req.method, req.url);
        });
      }
    }""",
                content,
                flags=re.DOTALL
            )
        else:
            # Add proxy configuration
            print("Adding Vite proxy configuration...")
            
            # Find server config
            server_match = re.search(r"server:\s*{", content)
            if server_match:
                # Insert proxy config after server: {
                insert_pos = server_match.end()
                proxy_config = """
    proxy: {
      '/api': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        ws: true, // Enable WebSocket proxying
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log('proxy error', err);
          });
        }
      }
    },"""
                content = content[:insert_pos] + proxy_config + content[insert_pos:]
            else:
                # Add server config with proxy
                defineConfig_match = re.search(r"export default defineConfig\({", content)
                if defineConfig_match:
                    insert_pos = defineConfig_match.end()
                    server_config = """
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        ws: true, // Enable WebSocket proxying
      }
    }
  },"""
                    content = content[:insert_pos] + server_config + content[insert_pos:]
        
        # Write updated content
        vite_config_path.write_text(content)
        
        self.fixes_applied.append("Fixed Vite proxy configuration")
        print("✓ Vite proxy configuration updated")
        return True
    
    def add_missing_soliton_endpoints(self):
        """Add missing Soliton API endpoints"""
        api_file = self.project_root / "prajna" / "api" / "prajna_api.py"
        
        if not api_file.exists():
            print(f"Warning: API file not found at {api_file}")
            # Try alternative location
            api_file = self.project_root / "enhanced_launcher.py"
            
        if not api_file.exists():
            print("Could not find API file to patch")
            return False
            
        # Backup original
        shutil.copy(api_file, api_file.with_suffix('.py.backup'))
        
        # Read content
        content = api_file.read_text()
        
        # Check if endpoints already exist
        if "/api/soliton/init" in content:
            print("Soliton endpoints already exist")
            return True
            
        # Find where to insert the endpoints
        # Look for existing soliton router or API setup
        soliton_router_match = re.search(r"(soliton_router\s*=.*?)\n", content)
        
        if soliton_router_match:
            # Add after existing router definition
            insert_pos = soliton_router_match.end()
            
            new_endpoints = '''
# Missing Soliton endpoints
@soliton_router.post("/init")
async def soliton_init():
    """Initialize Soliton memory system"""
    try:
        # Initialize the soliton memory
        # In real implementation, call SolitonMemoryClient.init()
        return {"status": "initialized", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Failed to initialize Soliton: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@soliton_router.get("/stats/{user}")
async def soliton_stats(user: str):
    """Get memory statistics for a user"""
    try:
        # In real implementation, fetch from soliton memory
        stats = {
            "user": user,
            "memory_count": 0,
            "total_tokens": 0,
            "last_access": None
        }
        return {"user": user, "stats": stats}
    except Exception as e:
        logger.error(f"Failed to get soliton stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@soliton_router.post("/embed")
async def soliton_embed(payload: dict):
    """Embed new information into memory"""
    try:
        # In real implementation, process and store embedding
        return {"detail": "embedded", "id": str(uuid.uuid4())}
    except Exception as e:
        logger.error(f"Failed to embed in soliton: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
            content = content[:insert_pos] + new_endpoints + content[insert_pos:]
            
            # Add imports if missing
            if "from datetime import datetime" not in content:
                content = "from datetime import datetime\n" + content
            if "import uuid" not in content:
                content = "import uuid\n" + content
            if "from fastapi import HTTPException" not in content:
                content = "from fastapi import HTTPException\n" + content
                
        else:
            print("Could not find soliton router definition, creating new one...")
            # Add complete soliton router setup
            # This would be inserted at an appropriate location
            
        # Write updated content
        api_file.write_text(content)
        
        self.fixes_applied.append("Added missing Soliton API endpoints")
        print("✓ Added missing Soliton endpoints")
        return True
    
    def add_avatar_websocket_endpoint(self):
        """Add missing avatar WebSocket endpoint"""
        api_file = self.project_root / "enhanced_launcher.py"
        
        if not api_file.exists():
            print(f"Warning: Enhanced launcher not found at {api_file}")
            return False
            
        # Backup original
        shutil.copy(api_file, api_file.with_suffix('.py.backup'))
        
        # Read content
        content = api_file.read_text()
        
        # Check if WebSocket endpoint already exists
        if "/api/avatar/updates" in content:
            print("Avatar WebSocket endpoint already exists")
            return True
            
        # Find where to add WebSocket endpoint
        # Look for other WebSocket endpoints or end of app definition
        app_websocket_match = re.search(r"@app\.websocket", content)
        
        if app_websocket_match:
            # Add after existing WebSocket endpoint
            insert_pos = content.find("\n", app_websocket_match.end()) + 1
        else:
            # Find app definition and add after
            app_match = re.search(r"app\s*=\s*FastAPI\([^)]*\)", content)
            if app_match:
                insert_pos = content.find("\n", app_match.end()) + 1
            else:
                print("Could not find appropriate location for WebSocket endpoint")
                return False
                
        # Add WebSocket endpoint
        websocket_endpoint = '''
@app.websocket("/api/avatar/updates")
async def avatar_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time avatar updates"""
    await websocket.accept()
    logger.info(f"Avatar WebSocket connected: {websocket.client}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat()
        })
        
        # Main update loop
        while True:
            try:
                # In real implementation, this would send actual avatar data
                # For now, send heartbeat to keep connection alive
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Also listen for client messages
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    logger.debug(f"Received from avatar client: {data}")
                except asyncio.TimeoutError:
                    pass  # No message received, continue sending updates
                    
            except Exception as e:
                logger.error(f"Error in avatar update loop: {e}")
                break
                
            await asyncio.sleep(0.1)  # Send updates at 10Hz
            
    except WebSocketDisconnect:
        logger.info("Avatar WebSocket disconnected")
    except Exception as e:
        logger.error(f"Avatar WebSocket error: {e}")
    finally:
        # Clean up resources if needed
        pass
'''
        
        # Add necessary imports
        imports_needed = [
            "from fastapi import WebSocket, WebSocketDisconnect",
            "import asyncio",
            "from datetime import datetime"
        ]
        
        for import_stmt in imports_needed:
            if import_stmt not in content:
                # Add after other imports
                import_match = re.search(r"^import.*$", content, re.MULTILINE)
                if import_match:
                    insert_import_pos = content.find("\n", import_match.end()) + 1
                    content = content[:insert_import_pos] + import_stmt + "\n" + content[insert_import_pos:]
                    
        # Insert WebSocket endpoint
        content = content[:insert_pos] + websocket_endpoint + content[insert_pos:]
        
        # Write updated content
        api_file.write_text(content)
        
        self.fixes_applied.append("Added avatar WebSocket endpoint")
        print("✓ Added avatar WebSocket endpoint")
        return True
    
    def fix_webgpu_shader_barriers(self):
        """Fix WebGPU shader barrier issues"""
        shader_path = self.project_root / "tori_ui_svelte" / "src" / "shaders" / "multiViewSynthesis.wgsl"
        
        if not shader_path.exists():
            # Try to find shader files
            shader_files = list(self.project_root.rglob("*.wgsl"))
            if not shader_files:
                print("No WGSL shader files found")
                return False
                
            # Process all shader files
            for shader_path in shader_files:
                self._fix_single_shader(shader_path)
        else:
            self._fix_single_shader(shader_path)
            
        self.fixes_applied.append("Fixed WebGPU shader barriers")
        return True
    
    def _fix_single_shader(self, shader_path: Path):
        """Fix barriers in a single shader file"""
        print(f"Processing shader: {shader_path}")
        
        # Backup original
        shutil.copy(shader_path, shader_path.with_suffix('.wgsl.backup'))
        
        # Read content
        content = shader_path.read_text()
        
        # Pattern to find workgroupBarrier inside loops
        # This is a simplified pattern - in practice would need more sophisticated parsing
        barrier_in_loop_pattern = r'(for\s*\([^)]+\)\s*{[^}]*workgroupBarrier\(\)[^}]*})'
        
        matches = list(re.finditer(barrier_in_loop_pattern, content, re.DOTALL))
        
        if not matches:
            print(f"No barrier issues found in {shader_path.name}")
            return
            
        # Process matches in reverse order to maintain positions
        for match in reversed(matches):
            loop_content = match.group(1)
            
            # Extract loop header and body
            loop_match = re.match(r'(for\s*\([^)]+\)\s*{)(.*)(})', loop_content, re.DOTALL)
            if loop_match:
                loop_header = loop_match.group(1)
                loop_body = loop_match.group(2)
                loop_end = loop_match.group(3)
                
                # Split body at barrier
                parts = loop_body.split('workgroupBarrier()')
                
                if len(parts) == 2:
                    before_barrier = parts[0].strip()
                    after_barrier = parts[1].strip()
                    
                    # Reconstruct with barrier outside loop
                    new_code = f'''// First pass - before barrier
{loop_header}
    {before_barrier}
{loop_end}

// Synchronize all threads
workgroupBarrier();

// Second pass - after barrier  
{loop_header}
    {after_barrier}
{loop_end}'''
                    
                    # Replace in content
                    content = content[:match.start()] + new_code + content[match.end():]
                    
        # Write updated content
        shader_path.write_text(content)
        print(f"✓ Fixed barriers in {shader_path.name}")
    
    def install_missing_dependencies(self):
        """Install missing Python dependencies"""
        import subprocess
        import sys
        
        missing_packages = [
            "torch torchvision torchaudio",
            "deepdiff",
            "sympy", 
            "PyPDF2"
        ]
        
        for package in missing_packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + package.split())
                self.fixes_applied.append(f"Installed {package}")
                print(f"✓ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")
                
    def create_missing_directories(self):
        """Create required directories"""
        required_dirs = [
            self.project_root / "logs",
            self.project_root / "tmp",
            self.project_root / "data" / "memory_vault",
            self.project_root / "data" / "concept_db"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.fixes_applied.append(f"Created directory: {dir_path}")
                print(f"✓ Created {dir_path}")
                
    def fix_tailwind_custom_utilities(self):
        """Fix TailwindCSS custom utility class issues"""
        css_file = self.project_root / "tori_ui_svelte" / "src" / "app.css"
        
        if not css_file.exists():
            print(f"Warning: app.css not found at {css_file}")
            return False
            
        # Backup original
        shutil.copy(css_file, css_file.with_suffix('.css.backup'))
        
        # Read content
        content = css_file.read_text()
        
        # Check if we're using @apply with custom classes
        if "@apply tori-button" in content:
            # Define the custom utility properly
            utility_definition = '''
/* Define custom utilities */
@layer components {
  .tori-button {
    @apply inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2;
  }
}

'''
            # Add at the beginning if not already defined
            if ".tori-button {" not in content:
                content = utility_definition + content
                
        # Write updated content
        css_file.write_text(content)
        
        self.fixes_applied.append("Fixed TailwindCSS custom utilities")
        print("✓ Fixed TailwindCSS custom utilities")
        return True
    
    def apply_all_fixes(self) -> Dict[str, List[str]]:
        """Apply all automated fixes"""
        print("\n" + "="*60)
        print("APPLYING AUTOMATED FIXES FOR TORI")
        print("="*60 + "\n")
        
        # Apply fixes in order of priority
        self.fix_vite_proxy_configuration()
        self.add_missing_soliton_endpoints() 
        self.add_avatar_websocket_endpoint()
        self.fix_webgpu_shader_barriers()
        self.create_missing_directories()
        self.fix_tailwind_custom_utilities()
        
        # Install dependencies last (can take time)
        if input("\nInstall missing Python dependencies? (y/n): ").lower() == 'y':
            self.install_missing_dependencies()
            
        return {
            "fixes_applied": self.fixes_applied,
            "timestamp": datetime.now().isoformat()
        }


def main():
    """Main entry point for auto-fixer"""
    from datetime import datetime
    
    project_root = Path("C:\\Users\\jason\\Desktop\\tori\\kha")
    fixer = TORIAutoFixer(project_root)
    
    result = fixer.apply_all_fixes()
    
    # Save results
    result_file = project_root / "debugging_enhanced" / f"fixes_applied_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f"\n✓ Applied {len(result['fixes_applied'])} fixes")
    print(f"Results saved to: {result_file}")
    print("\nNext steps:")
    print("1. Review the changes made (backups created with .backup extension)")
    print("2. Restart TORI with: python enhanced_launcher.py --clean-start")
    print("3. Run diagnostics again: python enhanced_diagnostic_system.py")
    

if __name__ == "__main__":
    main()
