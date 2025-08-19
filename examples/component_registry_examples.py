"""
Example: Using Component Registry with existing TORI components
Shows the minimal changes needed to add proper readiness tracking
"""

from python.core.component_registry import component_registry, track_initialization
from datetime import datetime


# Method 1: Manual registration (for existing classes)
def integrate_with_cognitive_engine():
    """Add to CognitiveEngine.__init__ at the end"""
    # At the start of __init__:
    component_registry.register_component('cognitive_engine')
    
    # ... existing initialization code ...
    
    # At the end when everything is ready:
    component_registry.mark_ready('cognitive_engine', {
        'initialized_at': datetime.now().isoformat(),
        'vector_dim': self.vector_dim
    })


# Method 2: Using the decorator (for new components)
@track_initialization('quantum_lattice', dependencies={'concept_mesh', 'cognitive_engine'})
class QuantumLattice:
    def __init__(self):
        # The decorator automatically:
        # 1. Registers this component
        # 2. Marks it ready if __init__ completes
        # 3. Marks it failed if __init__ raises an exception
        
        self.oscillators = self._initialize_oscillators()
        self.couplings = self._setup_couplings()
        # No need to call mark_ready() - decorator handles it!


# Method 3: For async components
class AsyncComponent:
    def __init__(self):
        component_registry.register_component('async_component')
        # Don't mark ready yet - we're not initialized
    
    async def initialize(self):
        """Async initialization"""
        try:
            await self._load_models()
            await self._connect_to_services()
            
            # NOW we're ready
            component_registry.mark_ready('async_component')
            
        except Exception as e:
            component_registry.mark_failed('async_component', str(e))
            raise


# Method 4: For external processes (like MCP server)
def monitor_external_process():
    """Run this in a thread to monitor external component"""
    component_registry.register_component('mcp_server', required=True)
    
    # Poll the external process
    while True:
        if check_mcp_health():  # Your existing health check
            component_registry.mark_ready('mcp_server', {
                'pid': mcp_process.pid,
                'port': 8100
            })
            break
        time.sleep(1)


# Method 5: Quick integration for all Python components
def quick_integrate_all_components():
    """
    Add this to your main startup to quickly integrate existing components
    without modifying their code (temporary solution)
    """
    
    # Import all components
    from python.core import CognitiveEngine, UnifiedMemoryVault, ConceptMesh
    
    # Monkey-patch readiness tracking
    original_cognitive_init = CognitiveEngine.__init__
    def tracked_cognitive_init(self, *args, **kwargs):
        component_registry.register_component('cognitive_engine')
        result = original_cognitive_init(self, *args, **kwargs)
        component_registry.mark_ready('cognitive_engine')
        return result
    CognitiveEngine.__init__ = tracked_cognitive_init
    
    # Repeat for other components...


# Example: How the launcher uses this
def enhanced_launcher_with_registry():
    """
    Replace timeout-based waiting with proper readiness checks
    """
    # Start all your services
    start_api_server()
    start_mcp_server()
    start_frontend()
    
    # Wait for readiness using the registry
    if component_registry.wait_for_ready(timeout=120):
        print("🎉 TORI fully initialized!")
        
        # Get detailed report
        report = component_registry.get_readiness_report()
        for name, info in report['components'].items():
            print(f"  ✅ {name}: initialized in {info['initialization_time']:.1f}s")
    else:
        print("⚠️ Some components failed to initialize")
        
        # Show what failed
        report = component_registry.get_readiness_report()
        for name, info in report['components'].items():
            if info['status'] == 'failed':
                print(f"  ❌ {name}: {info['error']}")


# Example: Gradual rollout
"""
You don't have to convert everything at once! Start with:

1. Add the registry to your API (readiness_routes.py)
2. Update your launcher to check /api/system/ready
3. Gradually add component_registry calls to each component
4. Components without registry calls will just not appear in the report

This way you get immediate benefit and can improve over time.
"""
