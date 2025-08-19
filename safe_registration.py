"""
Safe registration wrapper to prevent duplicates
"""

import logging

logger = logging.getLogger(__name__)

class SafeRegistrationMixin:
    """Mixin to add safe registration to any manager"""
    
    def safe_register(self, name: str, item: any, force: bool = False):
        """Register item only if it doesn't exist (unless force=True)"""
        if hasattr(self, 'exists') and self.exists(name) and not force:
            logger.debug(f"Skipping duplicate registration: {name}")
            return False
        
        # Call original register method
        if hasattr(self, 'register'):
            self.register(name, item)
            return True
        else:
            raise AttributeError(f"{self.__class__.__name__} has no register method")
    
    def register_once(self, name: str, item: any):
        """Register item only if it doesn't exist (never force)"""
        return self.safe_register(name, item, force=False)

def patch_managers():
    """Monkey patch managers to add safe registration"""
    try:
        # Patch tool manager
        from mcp_metacognitive.tools.tool_manager import ToolManager
        if not hasattr(ToolManager, 'safe_register'):
            ToolManager.__bases__ = (SafeRegistrationMixin,) + ToolManager.__bases__
            logger.info("Patched ToolManager with safe registration")
    except:
        pass
    
    try:
        # Patch resource manager
        from mcp_metacognitive.resources.resource_manager import ResourceManager
        if not hasattr(ResourceManager, 'safe_register'):
            ResourceManager.__bases__ = (SafeRegistrationMixin,) + ResourceManager.__bases__
            logger.info("Patched ResourceManager with safe registration")
    except:
        pass
    
    try:
        # Patch prompt manager
        from mcp_metacognitive.prompts.manager import PromptManager
        if not hasattr(PromptManager, 'safe_register'):
            PromptManager.__bases__ = (SafeRegistrationMixin,) + PromptManager.__bases__
            logger.info("Patched PromptManager with safe registration")
    except:
        pass

# Auto-patch on import
patch_managers()
