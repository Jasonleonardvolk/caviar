#!/usr/bin/env python3
"""
Patch to update enhanced_launcher.py with TORI/Saigon v5 integration
Run this to add v5 component support to existing launcher
"""

import sys
from pathlib import Path

def apply_v5_patch():
    """Apply v5 integration patch to enhanced_launcher.py"""
    
    launcher_path = Path("enhanced_launcher.py")
    if not launcher_path.exists():
        print("❌ enhanced_launcher.py not found")
        return False
    
    # Read existing file
    with open(launcher_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if "TORI/Saigon v5 Components" in content:
        print("✅ Already patched with v5 support")
        return True
    
    # Find import section
    import_marker = "# Optional MCP bridge import"
    if import_marker not in content:
        import_marker = "import traceback"
    
    # v5 imports to add
    v5_imports = '''
# ============================================================================
# TORI/Saigon v5 Components
# ============================================================================
try:
    from python.core.saigon_inference_v5 import SaigonInference
    from python.core.adapter_loader_v5 import MetadataManager
    from python.core.concept_mesh_v5 import MeshManager
    from python.core.user_context import UserContextManager
    from python.core.conversation_manager import ConversationManager
    from python.core.lattice_morphing import LatticeMorpher
    TORI_V5_AVAILABLE = True
except ImportError:
    TORI_V5_AVAILABLE = False
    print("ℹ️ TORI v5 components not available (optional)")

try:
    from hott_integration.psi_morphon import ConceptSynthesizer
    HOTT_AVAILABLE = True
except ImportError:
    HOTT_AVAILABLE = False

'''
    
    # Insert v5 imports after existing imports
    import_pos = content.find(import_marker)
    if import_pos > 0:
        # Find end of line
        newline_pos = content.find('\n', import_pos)
        content = content[:newline_pos+1] + v5_imports + content[newline_pos+1:]
    
    # Add v5 initialization in __init__
    init_addition = '''
        # TORI/Saigon v5 Components
        self.saigon_inference = None
        self.adapter_manager = None
        self.mesh_manager_v5 = None
        self.user_context_manager = None
        self.conversation_manager = None
        self.lattice_morpher = None
        self.concept_synthesizer = None
'''
    
    # Find __init__ method
    init_marker = "self.enhanced_logger = EnhancedLogger()"
    if init_marker in content:
        init_pos = content.find(init_marker)
        newline_pos = content.find('\n', init_pos)
        content = content[:newline_pos+1] + init_addition + content[newline_pos+1:]
    
    # Add v5 initialization method
    v5_init_method = '''
    def initialize_v5_components(self):
        """Initialize TORI/Saigon v5 components if available"""
        if not TORI_V5_AVAILABLE:
            return False
        
        try:
            self.logger.info("Initializing TORI v5 components...")
            
            # Core v5 components
            self.saigon_inference = SaigonInference()
            self.adapter_manager = MetadataManager()
            self.mesh_manager_v5 = MeshManager()
            self.user_context_manager = UserContextManager()
            
            # Conversation manager
            self.conversation_manager = ConversationManager(
                self.saigon_inference,
                self.mesh_manager_v5
            )
            
            # Lattice morphing
            self.lattice_morpher = LatticeMorpher()
            
            # HoTT synthesis if available
            if HOTT_AVAILABLE:
                self.concept_synthesizer = ConceptSynthesizer(
                    self.mesh_manager_v5,
                    self.lattice_morpher
                )
                self.logger.info("   ✅ HoTT/Psi-Morphon synthesis ready")
            
            self.logger.info("   ✅ TORI v5 components initialized")
            self.logger.info("   ✅ Multi-user inference ready")
            self.logger.info("   ✅ LoRA adapter hot-swapping ready")
            self.logger.info("   ✅ Continuous learning enabled")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize v5: {e}")
            return False
'''
    
    # Add method before launch method
    launch_marker = "def launch(self)"
    if launch_marker in content:
        launch_pos = content.find(launch_marker)
        content = content[:launch_pos] + v5_init_method + "\n    " + content[launch_pos:]
    
    # Add v5 initialization call in launch method
    v5_init_call = '''
        # Initialize TORI v5 components if available
        v5_initialized = False
        if TORI_V5_AVAILABLE:
            v5_initialized = self.initialize_v5_components()
'''
    
    # Find spot in launch method
    launch_start = "# Initialize core components"
    if launch_start in content:
        start_pos = content.find(launch_start)
        content = content[:start_pos] + v5_init_call + "\n        " + content[start_pos:]
    
    # Add v5 status display
    v5_status = '''
        # TORI v5 Status
        if v5_initialized:
            self.logger.info("\\n🚀 TORI/SAIGON v5 COMPONENTS:")
            if self.saigon_inference:
                self.logger.info("   ✅ Saigon Inference Engine: Active (multi-user, LoRA adapters)")
            if self.adapter_manager:
                self.logger.info("   ✅ Adapter Manager: Active (hot-swapping enabled)")
            if self.mesh_manager_v5:
                self.logger.info("   ✅ Mesh Manager v5: Active (evolution tracking)")
            if self.conversation_manager:
                self.logger.info("   ✅ Conversation Manager: Active (intent gap detection)")
            if self.lattice_morpher:
                self.logger.info("   ✅ Lattice Morphing: Active (AV sync ready)")
            if self.concept_synthesizer:
                self.logger.info("   ✅ HoTT/Psi-Morphon: Active (concept synthesis)")
            
            # Show v5 API endpoints
            self.logger.info("\\n🌐 TORI v5 API ENDPOINTS:")
            self.logger.info(f"   POST /api/saigon/infer - Multi-user inference")
            self.logger.info(f"   POST /api/saigon/adapters/hot-swap - Hot-swap adapters")
            self.logger.info(f"   GET /api/saigon/mesh/{{user_id}} - Get user mesh")
            self.logger.info(f"   POST /api/saigon/conversation/chat - Chat with intent tracking")
            self.logger.info(f"   POST /api/saigon/morph - Lattice morphing")
'''
    
    # Add before system info
    system_info_marker = "# System info"
    if system_info_marker in content:
        info_pos = content.find(system_info_marker)
        content = content[:info_pos] + v5_status + "\n        " + content[info_pos:]
    
    # Save backup
    backup_path = Path("enhanced_launcher.py.backup_v5")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Write patched file
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully patched enhanced_launcher.py with v5 support")
    print(f"📦 Backup saved to: {backup_path}")
    
    return True

def main():
    """Apply the patch"""
    print("\n🔧 TORI/Saigon v5 Integration Patch")
    print("="*50)
    
    if apply_v5_patch():
        print("\n✅ Patch applied successfully!")
        print("\n📝 New features added:")
        print("   • Saigon Inference Engine v5")
        print("   • Multi-user adapter management")
        print("   • Conversation manager with intent gaps")
        print("   • Lattice morphing with AV sync")
        print("   • HoTT/Psi-Morphon synthesis")
        print("\n🚀 Run enhanced_launcher.py to use v5 components!")
    else:
        print("\n❌ Patch failed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
