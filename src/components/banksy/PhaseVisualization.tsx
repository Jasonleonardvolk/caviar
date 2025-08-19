/**
 * Phase Visualization Component
 * 
 * Real-time visualization of Banksy oscillator phases using canvas-based rendering.
 * Shows phase evolution, order parameter, and synchronization patterns.
 */

import React, { useRef, useEffect, useState } from 'react';
import { BanksyState } from './BanksyApiClient';

interface PhaseVisualizationProps {
  state: BanksyState | null;
  width?: number;
  height?: number;
  className?: string;
}

const PhaseVisualization: React.FC<PhaseVisualizationProps> = ({
  state,
  width = 400,
  height = 300,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [history, setHistory] = useState<BanksyState[]>([]);
  const maxHistoryLength = 200; // Keep last 200 states

  // Update history when state changes
  useEffect(() => {
    if (state) {
      setHistory(prev => {
        const newHistory = [...prev, state];
        return newHistory.slice(-maxHistoryLength); // Keep only recent history
      });
    }
  }, [state]);

  // Clear history when starting new simulation
  useEffect(() => {
    if (state?.step === 1) {
      setHistory([state]);
    }
  }, [state?.step]);

  // Render visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !state) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#1f2937'; // Dark background
    ctx.fillRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.35;

    // Draw coordinate system
    drawCoordinateSystem(ctx, centerX, centerY, radius);

    // Draw phase circle
    drawPhaseCircle(ctx, centerX, centerY, radius);

    // Draw oscillator phases as points on the circle
    if (state.active_concepts) {
      drawOscillatorPhases(ctx, centerX, centerY, radius, state);
    }

    // Draw order parameter vector
    drawOrderParameter(ctx, centerX, centerY, radius, state);

    // Draw synchronization indicator
    drawSynchronizationIndicator(ctx, state);

    // Draw time series if we have history
    if (history.length > 1) {
      drawTimeSeries(ctx, history, width, height);
    }

  }, [state, history, width, height]);

  const drawCoordinateSystem = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number) => {
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;

    // X and Y axes
    ctx.beginPath();
    ctx.moveTo(centerX - radius - 10, centerY);
    ctx.lineTo(centerX + radius + 10, centerY);
    ctx.moveTo(centerX, centerY - radius - 10);
    ctx.lineTo(centerX, centerY + radius + 10);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#9ca3af';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('0', centerX + radius + 15, centerY + 4);
    ctx.fillText('π/2', centerX, centerY - radius - 15);
    ctx.fillText('π', centerX - radius - 15, centerY + 4);
    ctx.fillText('3π/2', centerX, centerY + radius + 20);
  };

  const drawPhaseCircle = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number) => {
    // Main circle
    ctx.strokeStyle = '#4b5563';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.stroke();

    // Quarter markers
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 1;
    for (let i = 0; i < 4; i++) {
      const angle = (i * Math.PI) / 2;
      const x1 = centerX + Math.cos(angle) * (radius - 5);
      const y1 = centerY + Math.sin(angle) * (radius - 5);
      const x2 = centerX + Math.cos(angle) * (radius + 5);
      const y2 = centerY + Math.sin(angle) * (radius + 5);
      
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  };

  const drawOscillatorPhases = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number, state: BanksyState) => {
    const concepts = Object.entries(state.active_concepts);
    
    concepts.forEach(([conceptId, activation], index) => {
      // Convert concept index to phase (simplified mapping)
      const phase = (index / concepts.length) * 2 * Math.PI + state.mean_phase;
      
      const x = centerX + Math.cos(phase) * radius;
      const y = centerY + Math.sin(phase) * radius;
      
      // Color based on activation level
      const intensity = Math.round(activation * 255);
      const color = `rgb(${intensity}, ${Math.round(intensity * 0.8)}, ${Math.round(255 - intensity * 0.5)})`;
      
      // Draw oscillator point
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 4 + activation * 3, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw connection to center for highly active oscillators
      if (activation > 0.7) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });
  };

  const drawOrderParameter = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number, state: BanksyState) => {
    const orderMagnitude = state.order_parameter;
    const orderPhase = state.mean_phase;
    
    if (orderMagnitude > 0.01) {
      const endX = centerX + Math.cos(orderPhase) * radius * orderMagnitude;
      const endY = centerY + Math.sin(orderPhase) * radius * orderMagnitude;
      
      // Order parameter vector
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      
      // Arrowhead
      const arrowSize = 8;
      const arrowAngle = Math.atan2(endY - centerY, endX - centerX);
      
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - arrowSize * Math.cos(arrowAngle - Math.PI / 6),
        endY - arrowSize * Math.sin(arrowAngle - Math.PI / 6)
      );
      ctx.lineTo(
        endX - arrowSize * Math.cos(arrowAngle + Math.PI / 6),
        endY - arrowSize * Math.sin(arrowAngle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fill();
    }
  };

  const drawSynchronizationIndicator = (ctx: CanvasRenderingContext2D, state: BanksyState) => {
    // Top-left indicator
    const x = 10;
    const y = 20;
    
    ctx.fillStyle = '#f3f4f6';
    ctx.font = '14px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Order: ${state.order_parameter.toFixed(3)}`, x, y);
    ctx.fillText(`Step: ${state.step}`, x, y + 20);
    ctx.fillText(`Sync: ${state.n_effective}`, x, y + 40);
    
    // Color-coded order parameter bar
    const barWidth = 100;
    const barHeight = 8;
    const barX = x;
    const barY = y + 50;
    
    // Background
    ctx.fillStyle = '#374151';
    ctx.fillRect(barX, barY, barWidth, barHeight);
    
    // Progress
    const progressWidth = barWidth * state.order_parameter;
    const color = state.order_parameter > 0.7 ? '#10b981' : 
                  state.order_parameter > 0.3 ? '#f59e0b' : '#ef4444';
    ctx.fillStyle = color;
    ctx.fillRect(barX, barY, progressWidth, barHeight);
  };

  const drawTimeSeries = (ctx: CanvasRenderingContext2D, history: BanksyState[], width: number, height: number) => {
    if (history.length < 2) return;
    
    // Draw time series in bottom right
    const chartWidth = 150;
    const chartHeight = 60;
    const chartX = width - chartWidth - 10;
    const chartY = height - chartHeight - 10;
    
    // Background
    ctx.fillStyle = 'rgba(31, 41, 55, 0.8)';
    ctx.fillRect(chartX, chartY, chartWidth, chartHeight);
    
    // Border
    ctx.strokeStyle = '#4b5563';
    ctx.lineWidth = 1;
    ctx.strokeRect(chartX, chartY, chartWidth, chartHeight);
    
    // Plot order parameter over time
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    history.forEach((state, index) => {
      const x = chartX + (index / (history.length - 1)) * chartWidth;
      const y = chartY + chartHeight - (state.order_parameter * chartHeight);
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Label
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Order Parameter', chartX + 2, chartY - 2);
  };

  return (
    <div className={`relative ${className}`}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="border border-gray-600 rounded bg-gray-900"
      />
      
      {!state && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75 rounded">
          <div className="text-gray-400 text-center">
            <div className="text-4xl mb-2">🌀</div>
            <div>Start simulation to see phase visualization</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PhaseVisualization;
