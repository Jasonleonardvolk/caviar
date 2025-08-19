/**
 * holoObject.jsx - Lightwave-bound cognition anchor for TORI
 * Spectral concept visualization with tetrachromatic representation
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const HoloObject = ({ 
  conceptId, 
  wavelength = 550, // Default green
  intensity = 0.8,
  radius = 40,
  spectralSignature = {},
  children,
  className = "",
  ...props 
}) => {
  const canvasRef = useRef(null);
  const [glowPulse, setGlowPulse] = useState(0);

  // Convert wavelength to RGB (simplified tetrachromatic fallback)
  const wavelengthToRGB = (lambda) => {
    let r, g, b;
    
    if (lambda >= 380 && lambda < 440) {
      r = -(lambda - 440) / (440 - 380);
      g = 0.0;
      b = 1.0;
    } else if (lambda >= 440 && lambda < 490) {
      r = 0.0;
      g = (lambda - 440) / (490 - 440);
      b = 1.0;
    } else if (lambda >= 490 && lambda < 510) {
      r = 0.0;
      g = 1.0;
      b = -(lambda - 510) / (510 - 490);
    } else if (lambda >= 510 && lambda < 580) {
      r = (lambda - 510) / (580 - 510);
      g = 1.0;
      b = 0.0;
    } else if (lambda >= 580 && lambda < 645) {
      r = 1.0;
      g = -(lambda - 645) / (645 - 580);
      b = 0.0;
    } else if (lambda >= 645 && lambda <= 750) {
      r = 1.0;
      g = 0.0;
      b = 0.0;
    } else {
      r = 0.0;
      g = 0.0;
      b = 0.0;
    }

    // Intensity falloff at edges
    let factor = 1.0;
    if ((lambda >= 380 && lambda < 420) || (lambda > 700 && lambda <= 750)) {
      factor = 0.3 + 0.7 * (lambda < 420 ? (lambda - 380) / (420 - 380) : (750 - lambda) / (750 - 700));
    }

    return {
      r: Math.round(255 * r * factor * intensity),
      g: Math.round(255 * g * factor * intensity),
      b: Math.round(255 * b * factor * intensity)
    };
  };

  const { r, g, b } = wavelengthToRGB(wavelength);
  const baseColor = `rgb(${r}, ${g}, ${b})`;
  const glowColor = `rgba(${r}, ${g}, ${b}, 0.4)`;

  // Pulse effect based on phase coherence
  useEffect(() => {
    const interval = setInterval(() => {
      setGlowPulse(prev => (prev + 0.1) % (2 * Math.PI));
    }, 50);
    return () => clearInterval(interval);
  }, []);

  const pulseIntensity = 0.5 + 0.5 * Math.sin(glowPulse);

  // Render holographic field effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width = radius * 2;
    const height = canvas.height = radius * 2;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Create radial gradient for holographic glow
    const gradient = ctx.createRadialGradient(
      radius, radius, 0,
      radius, radius, radius
    );
    gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${0.8 * pulseIntensity})`);
    gradient.addColorStop(0.7, `rgba(${r}, ${g}, ${b}, ${0.3 * pulseIntensity})`);
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // Add interference pattern for holographic effect
    for (let i = 0; i < 5; i++) {
      const ringRadius = (radius / 5) * (i + 1) * pulseIntensity;
      ctx.beginPath();
      ctx.arc(radius, radius, ringRadius, 0, 2 * Math.PI);
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${0.2 * (1 - i / 5)})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [r, g, b, radius, pulseIntensity]);

  return (
    <motion.div
      className={`relative inline-block ${className}`}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      style={{
        filter: `drop-shadow(0 0 ${10 * pulseIntensity}px ${glowColor})`,
      }}
      data-concept-id={conceptId}
      data-wavelength={wavelength}
      {...props}
    >
      {/* Holographic field canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{
          width: radius * 2,
          height: radius * 2,
          left: -radius,
          top: -radius,
          zIndex: 0,
        }}
      />
      
      {/* Content */}
      <div 
        className="relative z-10"
        style={{
          color: baseColor,
          textShadow: `0 0 5px ${glowColor}`,
        }}
      >
        {children}
      </div>

      {/* Spectral metadata overlay */}
      {spectralSignature && Object.keys(spectralSignature).length > 0 && (
        <div className="absolute -top-2 -right-2 text-xs opacity-60">
          λ{wavelength}
        </div>
      )}
    </motion.div>
  );
};

export default HoloObject;