/**
 * QuoteCanvas.tsx - Poetic visual renderer for TORI's awakening moments
 * Displays resonant quotes with spectral signatures and mood colors
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import HoloObject from '../ui/holoObject.jsx';
import { Quote } from '../../services/QuoteBank';

interface QuoteCanvasProps {
  quote: Quote | null;
  moodColor?: string;
  ghostAuraColor?: string;
  isVisible?: boolean;
  onComplete?: () => void;
  className?: string;
}

const QuoteCanvas: React.FC<QuoteCanvasProps> = ({
  quote,
  moodColor,
  ghostAuraColor,
  isVisible = true,
  onComplete,
  className = ''
}) => {
  const [animationPhase, setAnimationPhase] = useState(0);
  const [glowPulse, setGlowPulse] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Animate glow pulse
  useEffect(() => {
    const interval = setInterval(() => {
      setGlowPulse(prev => (prev + 0.05) % (2 * Math.PI));
    }, 50);
    return () => clearInterval(interval);
  }, []);

  // Background starfield effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !isVisible) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    const height = canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    const stars: Array<{ x: number; y: number; brightness: number; twinkle: number }> = [];
    
    // Generate stars
    for (let i = 0; i < 50; i++) {
      stars.push({
        x: Math.random() * (width / window.devicePixelRatio),
        y: Math.random() * (height / window.devicePixelRatio),
        brightness: Math.random(),
        twinkle: Math.random() * 2 * Math.PI
      });
    }

    const animate = () => {
      ctx.clearRect(0, 0, width / window.devicePixelRatio, height / window.devicePixelRatio);
      
      // Draw stars
      stars.forEach((star, i) => {
        star.twinkle += 0.02;
        const twinkleIntensity = 0.3 + 0.7 * (Math.sin(star.twinkle) * 0.5 + 0.5);
        const brightness = star.brightness * twinkleIntensity;
        
        ctx.fillStyle = `rgba(255, 255, 255, ${brightness * 0.8})`;
        ctx.beginPath();
        ctx.arc(star.x, star.y, 1, 0, 2 * Math.PI);
        ctx.fill();
        
        // Add occasional shooting stars
        if (Math.random() < 0.001) {
          ctx.strokeStyle = `rgba(255, 255, 255, ${brightness * 0.3})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(star.x, star.y);
          ctx.lineTo(star.x + Math.random() * 50 - 25, star.y + Math.random() * 50 - 25);
          ctx.stroke();
        }
      });

      if (isVisible) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }, [isVisible]);

  if (!quote || !isVisible) return null;

  const wavelength = quote.wavelength || 550;
  const displayMoodColor = ghostAuraColor || moodColor || `hsl(${(wavelength - 380) / (750 - 380) * 360}, 70%, 60%)`;

  return (
    <AnimatePresence>
      <motion.div
        className={`fixed inset-0 z-40 flex items-center justify-center ${className}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 1.5 }}
      >
        {/* Background canvas for starfield */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ 
            background: 'radial-gradient(circle at center, rgba(15, 23, 42, 0.95) 0%, rgba(0, 0, 0, 0.98) 100%)',
          }}
        />

        {/* Quote content */}
        <motion.div
          className="relative max-w-4xl mx-auto px-8 text-center"
          initial={{ scale: 0.8, y: 50 }}
          animate={{ scale: 1, y: 0 }}
          transition={{ 
            duration: 2,
            type: "spring",
            stiffness: 100,
            damping: 15
          }}
        >
          {/* Main quote text */}
          <HoloObject
            conceptId={`quote-${quote.id}`}
            wavelength={wavelength}
            intensity={0.7 + 0.3 * Math.sin(glowPulse)}
            radius={60}
            className="mb-8"
          >
            <motion.blockquote
              className="text-2xl md:text-4xl font-serif leading-relaxed mb-8"
              style={{
                textShadow: `0 0 20px ${displayMoodColor}40`,
                color: displayMoodColor
              }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5, duration: 2 }}
            >
              "{quote.text}"
            </motion.blockquote>
          </HoloObject>

          {/* Author attribution */}
          {quote.author && (
            <motion.cite
              className="text-lg text-slate-400 font-light"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 2, duration: 1 }}
            >
              — {quote.author}
            </motion.cite>
          )}

          {/* Phase signature indicator */}
          {quote.phaseSignature && (
            <motion.div
              className="mt-6 text-sm text-slate-500 uppercase tracking-wider"
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.7 }}
              transition={{ delay: 2.5, duration: 1 }}
            >
              Phase: {quote.phaseSignature}
            </motion.div>
          )}

          {/* Spectral wavelength indicator */}
          <motion.div
            className="mt-2 text-xs text-slate-600"
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            transition={{ delay: 3, duration: 1 }}
          >
            λ {wavelength}nm
          </motion.div>

          {/* Interaction hint */}
          <motion.div
            className="mt-8 text-sm text-slate-500"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 4, duration: 1 }}
          >
            <motion.span
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              Press any key to continue...
            </motion.span>
          </motion.div>
        </motion.div>

        {/* Ambient particles */}
        <div className="absolute inset-0 pointer-events-none">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 rounded-full"
              style={{ backgroundColor: displayMoodColor }}
              animate={{
                x: [
                  Math.random() * window.innerWidth,
                  Math.random() * window.innerWidth,
                  Math.random() * window.innerWidth
                ],
                y: [
                  Math.random() * window.innerHeight,
                  Math.random() * window.innerHeight,
                  Math.random() * window.innerHeight
                ],
                opacity: [0, 0.6, 0],
                scale: [0, 1, 0]
              }}
              transition={{
                duration: 10 + Math.random() * 10,
                repeat: Infinity,
                delay: Math.random() * 5
              }}
            />
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default QuoteCanvas;