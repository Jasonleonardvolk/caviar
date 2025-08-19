/**
 * Banksy Oscillator Control Panel
 * 
 * Main UI component for controlling and monitoring Banksy oscillator simulations.
 * Connects to the ALAN backend via BanksyApiClient for real-time visualization.
 */

import React, { useState, useEffect } from 'react';
import { useBanksyApi, BanksyConfig, BanksyState } from './BanksyApiClient';

interface BanksyOscillatorPanelProps {
  className?: string;
  onStateChange?: (state: BanksyState | null) => void;
}

const BanksyOscillatorPanel: React.FC<BanksyOscillatorPanelProps> = ({
  className = '',
  onStateChange,
}) => {
  const { 
    isConnected, 
    currentState, 
    isRunning, 
    error, 
    startRealtime, 
    stop, 
    getStateHistory 
  } = useBanksyApi();

  // Configuration state
  const [config, setConfig] = useState<BanksyConfig>({
    n_oscillators: 32,
    run_steps: 200,
    spin_substeps: 8,
    coupling_type: 'modular',
  });

  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Notify parent of state changes
  useEffect(() => {
    onStateChange?.(currentState);
  }, [currentState, onStateChange]);

  const handleStart = async () => {
    try {
      await startRealtime(config);
    } catch (err) {
      console.error('Failed to start Banksy simulation:', err);
    }
  };

  const handleStop = () => {
    stop();
  };

  const handleConfigChange = (key: keyof BanksyConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const getConnectionStatus = () => {
    if (!isConnected) return { color: 'text-red-500', icon: '🔴', text: 'Backend Offline' };
    if (isRunning) return { color: 'text-green-500', icon: '🟢', text: 'Simulation Running' };
    return { color: 'text-yellow-500', icon: '🟡', text: 'Ready' };
  };

  const status = getConnectionStatus();
  const stateHistory = getStateHistory();

  return (
    <div className={`bg-gray-900 border border-gray-700 rounded-lg p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="text-2xl">🌀</div>
          <div>
            <h2 className="text-xl font-bold text-white">Banksy Oscillator</h2>
            <div className={`text-sm ${status.color} flex items-center space-x-1`}>
              <span>{status.icon}</span>
              <span>{status.text}</span>
            </div>
          </div>
        </div>
        
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-gray-400 hover:text-white transition-colors"
        >
          {showAdvanced ? '▼ Simple' : '▶ Advanced'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-900/50 border border-red-700 rounded text-red-200 text-sm">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Controls */}
      <div className="space-y-4 mb-6">
        {/* Basic Configuration */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-gray-300 mb-1">Oscillators</label>
            <input
              type="number"
              value={config.n_oscillators}
              onChange={(e) => handleConfigChange('n_oscillators', parseInt(e.target.value))}
              className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white"
              min="4"
              max="256"
              disabled={isRunning}
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-1">Coupling Type</label>
            <select
              value={config.coupling_type}
              onChange={(e) => handleConfigChange('coupling_type', e.target.value as BanksyConfig['coupling_type'])}
              className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white"
              disabled={isRunning}
            >
              <option value="modular">Modular (Two Communities)</option>
              <option value="uniform">Uniform (All-to-All)</option>
              <option value="random">Random Network</option>
            </select>
          </div>
        </div>

        {/* Advanced Configuration */}
        {showAdvanced && (
          <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-700">
            <div>
              <label className="block text-sm text-gray-300 mb-1">Simulation Steps</label>
              <input
                type="number"
                value={config.run_steps}
                onChange={(e) => handleConfigChange('run_steps', parseInt(e.target.value))}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white"
                min="10"
                max="1000"
                disabled={isRunning}
              />
            </div>
            
            <div>
              <label className="block text-sm text-gray-300 mb-1">Spin Substeps</label>
              <input
                type="number"
                value={config.spin_substeps}
                onChange={(e) => handleConfigChange('spin_substeps', parseInt(e.target.value))}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white"
                min="1"
                max="16"
                disabled={isRunning}
              />
            </div>
          </div>
        )}

        {/* Control Buttons */}
        <div className="flex space-x-3 pt-4">
          <button
            onClick={handleStart}
            disabled={!isConnected || isRunning}
            className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium py-2 px-4 rounded transition-colors"
          >
            {isRunning ? 'Running...' : 'Start Simulation'}
          </button>
          
          <button
            onClick={handleStop}
            disabled={!isRunning}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded transition-colors"
          >
            Stop
          </button>
        </div>
      </div>

      {/* Current State Display */}
      {currentState && (
        <div className="bg-gray-800 rounded-lg p-4 space-y-3">
          <h3 className="text-lg font-semibold text-white mb-3">Current State</h3>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Step:</span>
              <span className="text-white ml-2 font-mono">{currentState.step}</span>
            </div>
            
            <div>
              <span className="text-gray-400">Time:</span>
              <span className="text-white ml-2 font-mono">{currentState.time.toFixed(3)}s</span>
            </div>
            
            <div>
              <span className="text-gray-400">Order Parameter:</span>
              <span className={`ml-2 font-mono ${
                currentState.order_parameter > 0.7 ? 'text-green-400' : 
                currentState.order_parameter > 0.3 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {currentState.order_parameter.toFixed(3)}
              </span>
            </div>
            
            <div>
              <span className="text-gray-400">Synchronized:</span>
              <span className="text-white ml-2 font-mono">
                {currentState.n_effective}/{config.n_oscillators}
              </span>
            </div>
            
            <div>
              <span className="text-gray-400">Mean Phase:</span>
              <span className="text-white ml-2 font-mono">{currentState.mean_phase.toFixed(3)} rad</span>
            </div>
            
            {currentState.trs_loss !== undefined && (
              <div>
                <span className="text-gray-400">TRS Loss:</span>
                <span className={`ml-2 font-mono ${
                  currentState.trs_loss < 1e-3 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {currentState.trs_loss.toExponential(2)}
                </span>
              </div>
            )}
          </div>

          {/* Quick visualization of synchronization */}
          <div className="mt-4">
            <div className="text-gray-400 text-sm mb-2">Synchronization Progress:</div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(currentState.order_parameter * 100)}%` }}
              ></div>
            </div>
            <div className="text-xs text-gray-400 mt-1">
              {(currentState.order_parameter * 100).toFixed(1)}% synchronized
            </div>
          </div>
        </div>
      )}

      {/* History Summary */}
      {stateHistory.length > 0 && (
        <div className="mt-4 text-sm text-gray-400">
          <div>History: {stateHistory.length} states recorded</div>
          <div>
            Peak Order: {Math.max(...stateHistory.map(s => s.order_parameter)).toFixed(3)}
          </div>
        </div>
      )}

      {/* Connection Status Footer */}
      {!isConnected && (
        <div className="mt-4 p-3 bg-orange-900/50 border border-orange-700 rounded text-orange-200 text-sm">
          <strong>Note:</strong> Banksy backend not detected. Make sure the simulation API server is running on port 8000.
          <br />
          <code className="text-xs bg-gray-800 px-1 py-0.5 rounded mt-1 inline-block">
            python alan_backend/server/simulation_api.py
          </code>
        </div>
      )}
    </div>
  );
};

export default BanksyOscillatorPanel;
