#!/usr/bin/env python3
"""
Unit test for launch order - ensures API starts before frontend
Run with: pytest test_launch_order.py
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, call

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_launcher import EnhancedUnifiedToriLauncher

class TestLaunchOrder:
    """Test that the launcher starts components in correct order"""
    
    def test_api_starts_before_frontend(self, monkeypatch):
        """Verify API server starts and is healthy before frontend starts"""
        
        # Track call order
        call_order = []
        
        # Mock methods to track order
        def mock_start_api(self, port):
            call_order.append(('api_start', port))
        
        def mock_wait_health(self, port):
            call_order.append(('api_wait', port))
            return True  # Simulate successful health check
        
        def mock_start_frontend(self):
            call_order.append(('frontend_start', None))
            return True
        
        def mock_start_mcp(self):
            call_order.append(('mcp_start', None))
            return True
        
        def mock_configure_prajna(self):
            call_order.append(('prajna_config', None))
            return True
        
        # Create mocks for threading
        mock_thread = Mock()
        mock_thread.start = Mock()
        mock_thread.join = Mock()
        
        # Patch all the methods
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, '_run_api_server', mock_start_api)
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, '_wait_for_api_health', mock_wait_health)
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'start_frontend_services_enhanced', mock_start_frontend)
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'start_mcp_metacognitive_server', mock_start_mcp)
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'configure_prajna_integration_enhanced', mock_configure_prajna)
        
        # Patch other required methods
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'print_banner', Mock())
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'update_status', Mock())
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'find_available_port', Mock(return_value=8002))
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'secure_port_aggressively', Mock(return_value=8002))
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'start_core_python_components', Mock(return_value=True))
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'start_stability_components', Mock(return_value=True))
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'save_port_config', Mock())
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'print_complete_system_ready', Mock())
        
        # Patch threading.Thread
        monkeypatch.setattr('threading.Thread', Mock(return_value=mock_thread))
        
        # Create launcher instance
        launcher = EnhancedUnifiedToriLauncher()
        launcher.api_port = 8002
        
        # Mock the thread join to prevent blocking
        mock_thread.join = Mock(side_effect=KeyboardInterrupt())
        
        # Run launch
        try:
            launcher.launch()
        except KeyboardInterrupt:
            pass  # Expected from our mock
        
        # Verify order
        assert len(call_order) >= 3, f"Expected at least 3 calls, got {len(call_order)}"
        
        # Find positions
        api_pos = next(i for i, (name, _) in enumerate(call_order) if name == 'api_start')
        wait_pos = next(i for i, (name, _) in enumerate(call_order) if name == 'api_wait')
        frontend_pos = next(i for i, (name, _) in enumerate(call_order) if name == 'frontend_start')
        
        # Assert correct order
        assert api_pos < wait_pos, "API should start before health check"
        assert wait_pos < frontend_pos, "Health check should complete before frontend starts"
        
        # Verify API thread was started
        assert mock_thread.start.called, "API thread should be started"
        
        print(f"✅ Launch order correct: {[name for name, _ in call_order]}")
    
    def test_api_failure_prevents_frontend(self, monkeypatch):
        """Test that frontend doesn't start if API health check fails"""
        
        call_order = []
        
        def mock_wait_health(self, port):
            call_order.append('api_wait_failed')
            return False  # Simulate failed health check
        
        def mock_start_frontend(self):
            call_order.append('frontend_start')
            return True
        
        # Patch methods
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, '_wait_for_api_health', mock_wait_health)
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'start_frontend_services_enhanced', mock_start_frontend)
        
        # Patch other required methods
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'print_banner', Mock())
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'update_status', Mock())
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'find_available_port', Mock(return_value=8002))
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, 'secure_port_aggressively', Mock(return_value=8002))
        monkeypatch.setattr(EnhancedUnifiedToriLauncher, '_run_api_server', Mock())
        monkeypatch.setattr('threading.Thread', Mock())
        
        # Create launcher
        launcher = EnhancedUnifiedToriLauncher()
        launcher.api_port = 8002
        
        # Run launch - should return 1 on API failure
        result = launcher.launch()
        
        assert result == 1, "Launch should return 1 on API failure"
        assert 'api_wait_failed' in call_order, "API health check should be called"
        assert 'frontend_start' not in call_order, "Frontend should NOT start if API fails"
        
        print("✅ Frontend correctly blocked when API fails")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
