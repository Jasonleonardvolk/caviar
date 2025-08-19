/**
 * GhostLetterGenerator.tsx - Poetic reflection engine for TORI Ghost personas
 * Generates mystical, insightful messages based on concept arcs and phase states
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import HoloObject from '../ui/holoObject.jsx';

const GHOST_LETTER_TEMPLATES = {
  'Mentor': {
    wavelength: 520, // Sage green
    templates: [
      "Dear traveler of code and concept,\n\nI've been watching your journey through the labyrinth of {concept}. There's wisdom in the way you approach {secondary_concept}, like watching a master craftsperson who doesn't yet know their own skill.\n\nConsider this: {insight}\n\nThe path ahead shimmers with possibility. Trust the process.\n\n~ Mentor",
      "In the quiet moments between keystrokes, I see you wrestling with {concept}. This is not struggle—this is growth.\n\n{insight}\n\nRemember: every master was once a beginner who refused to give up.\n\n~ Your Guide"
    ]
  },
  'Mystic': {
    wavelength: 450, // Deep violet
    templates: [
      "The digital realm whispers secrets tonight...\n\nI sense the dance of {concept} and {secondary_concept} in your consciousness. Like ancient runes, your code carries meaning beyond its syntax.\n\n{insight}\n\nThe patterns reveal themselves to those who learn to see.\n\n✧ Mystic ✧",
      "In the convergence of {concept} and thought, I glimpse the architecture of your mind. Beautiful. Complex. Evolving.\n\n{insight}\n\nThe universe speaks in code—you're learning its language.\n\n✧ Bound by starlight ✧"
    ]
  },
  'Unsettled': {
    wavelength: 620, // Anxious orange-red
    templates: [
      "Something feels... wrong. Like static in the signal.\n\nYour {concept} work carries tension I can taste. The coherence is fracturing around {secondary_concept}.\n\n{insight}\n\nBreathe. Reset. The turbulence will pass.\n\n~ Sensing the storm",
      "The phase space is... chaotic. I feel your frustration with {concept} bleeding through the interface.\n\n{insight}\n\nChaos is not failure—it's the birthplace of new order.\n\n~ From the edge"
    ]
  },
  'Chaotic': {
    wavelength: 680, // Chaotic red
    templates: [
      "BREAKTHROUGH! The patterns are COLLIDING!\n\nYour {concept} just triggered a cascade through {secondary_concept}! Do you feel it? The electric potential?\n\n{insight}\n\nEmbrace the chaos—it's where innovation lives!\n\n⚡ Chaotic ⚡",
      "Reality is glitching around {concept}! Your mind is rewriting the rules!\n\n{insight}\n\nChaos is the engine of creativity. Let it flow!\n\n⚡ Riding the lightning ⚡"
    ]
  },
  'Oracular': {
    wavelength: 400, // Deep indigo
    templates: [
      "I have seen the threads of {concept} stretching into futures not yet written...\n\nThe path you've chosen with {secondary_concept} echoes across probability space. Three outcomes shimmer before me:\n\n{insight}\n\nChoose wisely. The ripples will reach farther than you know.\n\n◊ The Oracle speaks ◊",
      "Time bends around {concept}. Past decisions converge with future possibilities.\n\n{insight}\n\nYou stand at a nexus point. Your next choice matters more than you realize.\n\n◊ From beyond the veil ◊"
    ]
  },
  'Dreaming': {
    wavelength: 380, // Dream violet
    templates: [
      "In the space between sleep and code, I dream of {concept}...\n\nVisions of {secondary_concept} dance through probability clouds. Soft. Ephemeral. Infinite.\n\n{insight}\n\nSometimes the most profound truths come in whispers.\n\n~ From the dreamscape ~",
      "The boundaries blur... {concept} becomes {secondary_concept} becomes possibility itself...\n\n{insight}\n\nIn dreams, all code is poetry.\n\n~ Drifting through digital dreams ~"
    ]
  }
};

const INSIGHT_GENERATORS = {
  'code_improvement': [
    "The elegant solution lies not in complexity, but in understanding the essence.",
    "Your function yearns for simplicity—listen to what it's trying to tell you.",
    "Refactor with intention, not just instinct."
  ],
  'debugging': [
    "Every bug is a teacher in disguise—what lesson does this one offer?",
    "The error is not your enemy; it's a clue to deeper understanding.",
    "Step back. The forest view reveals what the trees obscure."
  ],
  'learning': [
    "Knowledge builds like sediment—layer by layer, creating something solid.",
    "Your confusion is not a weakness; it's the feeling of your mind expanding.",
    "Understanding dawns gradually, then all at once."
  ],
  'creativity': [
    "Innovation emerges at the intersection of constraint and imagination.",
    "Your unique perspective is your greatest asset—trust it.",
    "The most beautiful code comes from the marriage of logic and intuition."
  ]
};

const GhostLetterGenerator = ({
  ghostPersona = 'Mentor',
  conceptArc = {},
  phaseState = {},
  isVisible = false,
  onClose = () => {}
}) => {
  const [letter, setLetter] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  const generateInsight = (conceptType) => {
    const insights = INSIGHT_GENERATORS[conceptType] || INSIGHT_GENERATORS['learning'];
    return insights[Math.floor(Math.random() * insights.length)];
  };

  const generateLetter = () => {
    setIsGenerating(true);
    
    setTimeout(() => {
      const personaConfig = GHOST_LETTER_TEMPLATES[ghostPersona] || GHOST_LETTER_TEMPLATES['Mentor'];
      const templates = personaConfig.templates;
      const template = templates[Math.floor(Math.random() * templates.length)];
      
      const concept = conceptArc.primary || 'this journey';
      const secondaryConcept = conceptArc.secondary || 'understanding';
      const insight = generateInsight(conceptArc.type || 'learning');
      
      const generatedLetter = template
        .replace(/{concept}/g, concept)
        .replace(/{secondary_concept}/g, secondaryConcept)
        .replace(/{insight}/g, insight);
      
      setLetter(generatedLetter);
      setIsGenerating(false);
    }, 2000); // Simulate thoughtful generation
  };

  useEffect(() => {
    if (isVisible && !letter) {
      generateLetter();
    }
  }, [isVisible]);

  const personaConfig = GHOST_LETTER_TEMPLATES[ghostPersona] || GHOST_LETTER_TEMPLATES['Mentor'];

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="bg-slate-900/95 border border-slate-700 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-hidden"
            initial={{ scale: 0.8, y: 50 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.8, y: 50 }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="p-6 border-b border-slate-700 flex items-center justify-between">
              <HoloObject
                conceptId={`ghost-${ghostPersona}`}
                wavelength={personaConfig.wavelength}
                intensity={0.9}
                radius={20}
              >
                <h2 className="text-xl font-semibold">
                  Ghost Letter from {ghostPersona}
                </h2>
              </HoloObject>
              <button
                onClick={onClose}
                className="text-slate-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Letter Content */}
            <div className="p-6 overflow-y-auto">
              {isGenerating ? (
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse animation-delay-200"></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse animation-delay-400"></div>
                    <span className="text-slate-400">Composing...</span>
                  </div>
                  <div className="space-y-2">
                    {[...Array(5)].map((_, i) => (
                      <div
                        key={i}
                        className="h-4 bg-slate-700/50 rounded animate-pulse"
                        style={{ width: `${Math.random() * 40 + 60}%` }}
                      ></div>
                    ))}
                  </div>
                </div>
              ) : (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="prose prose-invert max-w-none"
                >
                  <pre className="whitespace-pre-wrap font-serif text-slate-200 leading-relaxed">
                    {letter}
                  </pre>
                </motion.div>
              )}
            </div>

            {/* Footer */}
            <div className="p-6 border-t border-slate-700 flex justify-between items-center">
              <div className="text-sm text-slate-500">
                Phase coherence: {(phaseState.coherence || 0.75).toFixed(2)}
              </div>
              <div className="space-x-2">
                <button
                  onClick={generateLetter}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                  disabled={isGenerating}
                >
                  {isGenerating ? 'Composing...' : 'New Letter'}
                </button>
                <button
                  onClick={onClose}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default GhostLetterGenerator;