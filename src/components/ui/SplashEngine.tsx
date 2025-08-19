/**
 * SplashEngine.tsx - TORI's awakening orchestrator
 * Manages boot-time splash experiences with contextual wisdom
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import QuoteCanvas from '../ui/QuoteCanvas';
import { QuoteBankService, Quote } from '../../services/QuoteBank';

interface SplashEngineProps {
  isVisible?: boolean;
  onComplete?: () => void;
  userMBTI?: string;
  phaseSignature?: string;
  conceptClusters?: string[];
  ghostAuraColor?: string;
  sessionContext?: {
    isFirstVisit?: boolean;
    lastActiveTime?: Date;
    recentActivity?: string[];
  };
}

interface SplashState {
  currentQuote: Quote | null;
  showLogo: boolean;
  showQuote: boolean;
  isComplete: boolean;
  animationPhase: 'logo' | 'quote' | 'complete';
}

const SplashEngine: React.FC<SplashEngineProps> = ({
  isVisible = false,
  onComplete,
  userMBTI,
  phaseSignature,
  conceptClusters = [],
  ghostAuraColor,
  sessionContext = {}
}) => {
  const [state, setState] = useState<SplashState>({
    currentQuote: null,
    showLogo: false,
    showQuote: false,
    isComplete: false,
    animationPhase: 'logo'
  });

  const quoteBankService = QuoteBankService.getInstance();

  // Determine context and select appropriate quote
  useEffect(() => {
    if (!isVisible) return;

    const timeContext = quoteBankService.getCurrentTimeContext();
    
    // Determine user context
    const getUserRecentConceptClusters = () => {
      // In production, this would fetch from concept mesh
      // For now, use provided clusters or defaults
      return conceptClusters.length > 0 
        ? conceptClusters 
        : ['learning', 'exploration', 'creativity'];
    };

    const recentClusters = getUserRecentConceptClusters();
    
    // Select contextual quote
    const selectedQuote = quoteBankService.selectQuote({
      mbtiType: userMBTI,
      phaseSignature: phaseSignature,
      timeOfDay: timeContext,
      conceptClusters: recentClusters,
      previousQuoteIds: [] // Could track in localStorage
    });

    setState(prev => ({
      ...prev,
      currentQuote: selectedQuote,
      showLogo: true
    }));

    // Sequence the splash animation
    const sequence = async () => {
      // Phase 1: Logo appearance (2s)
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Phase 2: Quote emergence (3s)
      setState(prev => ({
        ...prev,
        showQuote: true,
        animationPhase: 'quote'
      }));
      
      await new Promise(resolve => setTimeout(resolve, 4000));
      
      // Phase 3: Completion
      setState(prev => ({
        ...prev,
        isComplete: true,
        animationPhase: 'complete'
      }));

      // Auto-complete after total time if no interaction
      setTimeout(() => {
        if (onComplete) onComplete();
      }, 1000);
    };

    sequence();
  }, [isVisible, userMBTI, phaseSignature, conceptClusters]);

  // Handle user interaction to skip
  useEffect(() => {
    const handleKeyPress = () => {
      if (state.animationPhase === 'quote' && onComplete) {
        onComplete();
      }
    };

    const handleClick = () => {
      if (state.animationPhase === 'quote' && onComplete) {
        onComplete();
      }
    };

    if (isVisible) {
      window.addEventListener('keydown', handleKeyPress);
      window.addEventListener('click', handleClick);
    }

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
      window.removeEventListener('click', handleClick);
    };
  }, [isVisible, state.animationPhase, onComplete]);

  const getGreetingMessage = (): string => {
    const hour = new Date().getHours();
    const { isFirstVisit, lastActiveTime } = sessionContext;
    
    if (isFirstVisit) {
      return 'Welcome to TORI';
    }
    
    if (lastActiveTime) {
      const timeSinceActive = Date.now() - lastActiveTime.getTime();
      const hoursSince = timeSinceActive / (1000 * 60 * 60);
      
      if (hoursSince > 24) {
        return 'Welcome back, traveler';
      } else if (hoursSince > 8) {
        return 'Resuming your journey';
      } else {
        return 'Continuing where we left off';
      }
    }
    
    if (hour >= 5 && hour < 12) return 'Good morning';
    if (hour >= 12 && hour < 17) return 'Good afternoon';
    if (hour >= 17 && hour < 22) return 'Good evening';
    return 'Greetings, night wanderer';
  };

  if (!isVisible) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 bg-slate-900">
        {/* Logo Phase */}
        <AnimatePresence>
          {state.showLogo && (
            <motion.div
              className="absolute inset-0 flex items-center justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 1 }}
            >
              <div className="text-center">
                {/* TORI Logo */}
                <motion.div
                  className="mb-8"
                  initial={{ scale: 0.5, y: 20 }}
                  animate={{ scale: 1, y: 0 }}
                  transition={{ 
                    duration: 1.5,
                    type: "spring",
                    stiffness: 100
                  }}
                >
                  <div 
                    className="text-6xl md:text-8xl font-thin tracking-wider mb-4"
                    style={{
                      background: 'linear-gradient(45deg, #0ea5e9, #8b5cf6, #06b6d4)',
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      textShadow: '0 0 30px rgba(14, 165, 233, 0.3)'
                    }}
                  >
                    TORI
                  </div>
                  
                  <motion.div
                    className="text-lg text-slate-400 tracking-wider"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.8, duration: 1 }}
                  >
                    {getGreetingMessage()}
                  </motion.div>
                </motion.div>

                {/* Pulse rings */}
                <div className="relative">
                  {[...Array(3)].map((_, i) => (
                    <motion.div
                      key={i}
                      className="absolute inset-0 border border-blue-400/20 rounded-full"
                      animate={{
                        scale: [1, 2, 3],
                        opacity: [0.8, 0.4, 0]
                      }}
                      transition={{
                        duration: 3,
                        repeat: Infinity,
                        delay: i * 1
                      }}
                      style={{
                        width: '100px',
                        height: '100px',
                        left: '50%',
                        top: '50%',
                        transform: 'translate(-50%, -50%)'
                      }}
                    />
                  ))}
                </div>

                {/* Loading indicator */}
                <motion.div
                  className="mt-16 flex items-center justify-center space-x-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 1.5, duration: 0.5 }}
                >
                  <div className="text-sm text-slate-500">Awakening consciousness</div>
                  {[...Array(3)].map((_, i) => (
                    <motion.div
                      key={i}
                      className="w-1 h-1 bg-blue-400 rounded-full"
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        delay: i * 0.3
                      }}
                    />
                  ))}
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Quote Phase */}
        <AnimatePresence>
          {state.showQuote && state.currentQuote && (
            <QuoteCanvas
              quote={state.currentQuote}
              ghostAuraColor={ghostAuraColor}
              isVisible={true}
              onComplete={onComplete}
            />
          )}
        </AnimatePresence>

        {/* Context indicators */}
        {isVisible && (
          <div className="absolute bottom-8 left-8 text-xs text-slate-600 space-y-1">
            {userMBTI && <div>Type: {userMBTI}</div>}
            {phaseSignature && <div>Phase: {phaseSignature}</div>}
            {conceptClusters.length > 0 && (
              <div>Context: {conceptClusters.slice(0, 3).join(', ')}</div>
            )}
          </div>
        )}

        {/* Version/Build info */}
        <div className="absolute bottom-8 right-8 text-xs text-slate-600">
          TORI v0.9.0-Phase9
        </div>
      </div>
    </AnimatePresence>
  );
};

export default SplashEngine;