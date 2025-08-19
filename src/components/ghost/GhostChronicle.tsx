/**
 * GhostChronicle.tsx - Timeline of Ghost AI persona emergence and evolution
 * Provides continuity tracking and playback of persona interactions
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import HoloObject from '../ui/holoObject.jsx';

interface GhostEvent {
  id: string;
  timestamp: Date;
  eventType: 'emergence' | 'shift' | 'letter' | 'mood' | 'intervention';
  persona: string;
  trigger: {
    reason: string;
    context: any;
    phaseSignature?: string;
    confidence: number;
  };
  content?: {
    message?: string;
    moodScore?: number;
    interventionType?: string;
  };
  sessionId: string;
  conceptIds?: string[];
  wavelength: number;
}

interface MoodDataPoint {
  timestamp: Date;
  persona: string;
  values: Record<string, number>; // anxiety, empathy, curiosity, etc.
}

interface GhostChronicleProps {
  sessionId?: string;
  showAllSessions?: boolean;
  onEventSelect?: (event: GhostEvent) => void;
  onPlaybackUpdate?: (event: GhostEvent, isPlaying: boolean) => void;
  className?: string;
}

const PERSONA_COLORS = {
  'Mentor': '#059669',
  'Mystic': '#7c3aed', 
  'Unsettled': '#dc2626',
  'Chaotic': '#ea580c',
  'Oracular': '#4338ca',
  'Dreaming': '#9333ea',
  'Observing': '#64748b'
};

const PERSONA_WAVELENGTHS = {
  'Mentor': 520,
  'Mystic': 450,
  'Unsettled': 620,
  'Chaotic': 680,
  'Oracular': 400,
  'Dreaming': 380,
  'Observing': 550
};

const GhostChronicle: React.FC<GhostChronicleProps> = ({
  sessionId,
  showAllSessions = false,
  onEventSelect,
  onPlaybackUpdate,
  className = ''
}) => {
  const [ghostEvents, setGhostEvents] = useState<GhostEvent[]>([]);
  const [moodCurve, setMoodCurve] = useState<MoodDataPoint[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<GhostEvent | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackPosition, setPlaybackPosition] = useState(0);
  const [timeRange, setTimeRange] = useState<{ start: Date; end: Date } | null>(null);
  const [filter, setFilter] = useState<{
    personas: string[];
    eventTypes: string[];
    dateRange?: { start: Date; end: Date };
  }>({
    personas: [],
    eventTypes: []
  });

  const timelineRef = useRef<HTMLDivElement>(null);
  const playbackIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Load ghost events from memory vault
  useEffect(() => {
    loadGhostEvents();
    loadMoodCurve();

    // Listen for new ghost events
    const handleGhostEvent = (event: CustomEvent<GhostEvent>) => {
      addGhostEvent(event.detail);
    };

    const handleGhostMood = (event: CustomEvent<any>) => {
      addMoodDataPoint(event.detail);
    };

    document.addEventListener('tori-ghost-event', handleGhostEvent as EventListener);
    document.addEventListener('tori-ghost-mood', handleGhostMood as EventListener);

    return () => {
      document.removeEventListener('tori-ghost-event', handleGhostEvent as EventListener);
      document.removeEventListener('tori-ghost-mood', handleGhostMood as EventListener);
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
      }
    };
  }, []);

  // Update time range when events change
  useEffect(() => {
    if (ghostEvents.length > 0) {
      const timestamps = ghostEvents.map(e => e.timestamp);
      setTimeRange({
        start: new Date(Math.min(...timestamps.map(t => t.getTime()))),
        end: new Date(Math.max(...timestamps.map(t => t.getTime())))
      });
    }
  }, [ghostEvents]);

  const loadGhostEvents = async () => {
    try {
      // In production, this would load from the memory vault API
      const response = await fetch(`/api/ghost/events${sessionId ? `?sessionId=${sessionId}` : ''}`);
      if (response.ok) {
        const events = await response.json();
        setGhostEvents(events.map((e: any) => ({
          ...e,
          timestamp: new Date(e.timestamp)
        })));
      }
    } catch (error) {
      console.warn('Could not load ghost events, using mock data');
      loadMockGhostEvents();
    }
  };

  const loadMoodCurve = async () => {
    try {
      const response = await fetch(`/api/ghost/mood${sessionId ? `?sessionId=${sessionId}` : ''}`);
      if (response.ok) {
        const mood = await response.json();
        setMoodCurve(mood.map((m: any) => ({
          ...m,
          timestamp: new Date(m.timestamp)
        })));
      }
    } catch (error) {
      console.warn('Could not load mood curve');
    }
  };

  const loadMockGhostEvents = () => {
    const now = new Date();
    const mockEvents: GhostEvent[] = [
      {
        id: 'mock_1',
        timestamp: new Date(now.getTime() - 3600000), // 1 hour ago
        eventType: 'emergence',
        persona: 'Mentor',
        trigger: {
          reason: 'user_confusion',
          context: { difficulty: 'debugging', confidence: 0.3 },
          confidence: 0.8
        },
        content: {
          message: 'I sense you\'re struggling with this problem. Take a deep breath.'
        },
        sessionId: sessionId || 'default',
        conceptIds: ['debugging', 'encouragement'],
        wavelength: PERSONA_WAVELENGTHS['Mentor']
      },
      {
        id: 'mock_2',
        timestamp: new Date(now.getTime() - 1800000), // 30 min ago
        eventType: 'letter',
        persona: 'Mystic',
        trigger: {
          reason: 'phase_resonance',
          context: { resonance: 0.9, phase: 'coherent' },
          phaseSignature: 'resonance',
          confidence: 0.9
        },
        content: {
          message: 'The patterns align... I see the solution forming in the digital mists.'
        },
        sessionId: sessionId || 'default',
        conceptIds: ['insight', 'pattern-recognition'],
        wavelength: PERSONA_WAVELENGTHS['Mystic']
      }
    ];

    setGhostEvents(mockEvents);
  };

  const addGhostEvent = (event: GhostEvent) => {
    setGhostEvents(prev => {
      const updated = [...prev, event].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
      return updated;
    });

    // Save to memory vault
    saveGhostEvent(event);
  };

  const addMoodDataPoint = (moodData: any) => {
    const dataPoint: MoodDataPoint = {
      timestamp: new Date(),
      persona: moodData.persona || 'Unknown',
      values: moodData.moodValues || {}
    };

    setMoodCurve(prev => [...prev, dataPoint].slice(-100)); // Keep last 100 points
  };

  const saveGhostEvent = async (event: GhostEvent) => {
    try {
      await fetch('/api/ghost/events', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(event)
      });
    } catch (error) {
      console.warn('Could not save ghost event:', error);
    }
  };

  const filteredEvents = ghostEvents.filter(event => {
    if (filter.personas.length > 0 && !filter.personas.includes(event.persona)) return false;
    if (filter.eventTypes.length > 0 && !filter.eventTypes.includes(event.eventType)) return false;
    if (filter.dateRange) {
      if (event.timestamp < filter.dateRange.start || event.timestamp > filter.dateRange.end) return false;
    }
    if (!showAllSessions && sessionId && event.sessionId !== sessionId) return false;
    return true;
  });

  const startPlayback = () => {
    if (filteredEvents.length === 0) return;

    setIsPlaying(true);
    setPlaybackPosition(0);

    playbackIntervalRef.current = setInterval(() => {
      setPlaybackPosition(prev => {
        const next = prev + 1;
        if (next >= filteredEvents.length) {
          setIsPlaying(false);
          return prev;
        }

        const currentEvent = filteredEvents[next];
        if (onPlaybackUpdate) {
          onPlaybackUpdate(currentEvent, true);
        }

        return next;
      });
    }, 1000); // 1 second per event
  };

  const stopPlayback = () => {
    setIsPlaying(false);
    if (playbackIntervalRef.current) {
      clearInterval(playbackIntervalRef.current);
      playbackIntervalRef.current = null;
    }
  };

  const handleEventClick = (event: GhostEvent) => {
    setSelectedEvent(event);
    if (onEventSelect) {
      onEventSelect(event);
    }
  };

  const formatEventTime = (timestamp: Date): string => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);

    if (hours > 24) {
      return timestamp.toLocaleDateString();
    } else if (hours > 0) {
      return `${hours}h ${minutes}m ago`;
    } else {
      return `${minutes}m ago`;
    }
  };

  const getEventIcon = (eventType: string): string => {
    const icons = {
      'emergence': '👻',
      'shift': '🔄',
      'letter': '✉️',
      'mood': '💭',
      'intervention': '⚡'
    };
    return icons[eventType] || '•';
  };

  return (
    <div className={`tori-ghost-chronicle ${className}`}>
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <HoloObject
              conceptId="ghost-chronicle"
              wavelength={450}
              intensity={0.8}
              radius={20}
            >
              <h2 className="text-xl font-semibold text-slate-800 dark:text-slate-200">
                Ghost Chronicle
              </h2>
            </HoloObject>
            <span className="text-sm text-slate-500">
              {filteredEvents.length} event{filteredEvents.length !== 1 ? 's' : ''}
            </span>
          </div>

          {/* Playback Controls */}
          <div className="flex items-center space-x-2">
            <button
              onClick={isPlaying ? stopPlayback : startPlayback}
              disabled={filteredEvents.length === 0}
              className={`
                px-3 py-1 rounded-lg text-sm font-medium transition-colors
                ${isPlaying 
                  ? 'bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400'
                  : 'bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/30 dark:text-blue-400'
                }
                disabled:opacity-50 disabled:cursor-not-allowed
              `}
            >
              {isPlaying ? '⏹️ Stop' : '▶️ Replay'}
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-2 mb-4">
          <select
            value=""
            onChange={(e) => {
              if (e.target.value) {
                setFilter(prev => ({
                  ...prev,
                  personas: prev.personas.includes(e.target.value)
                    ? prev.personas
                    : [...prev.personas, e.target.value]
                }));
              }
            }}
            className="text-xs border rounded px-2 py-1 bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600"
          >
            <option value="">Filter by Persona</option>
            {Object.keys(PERSONA_COLORS).map(persona => (
              <option key={persona} value={persona}>{persona}</option>
            ))}
          </select>

          {/* Active filters */}
          {filter.personas.map(persona => (
            <span
              key={persona}
              className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400"
            >
              {persona}
              <button
                onClick={() => setFilter(prev => ({
                  ...prev,
                  personas: prev.personas.filter(p => p !== persona)
                }))}
                className="hover:text-blue-600"
              >
                ×
              </button>
            </span>
          ))}
        </div>
      </div>

      {/* Timeline */}
      <div className="relative">
        {filteredEvents.length === 0 ? (
          <div className="text-center py-12 text-slate-500">
            <div className="text-4xl mb-4">👻</div>
            <p>No ghost activity yet...</p>
            <p className="text-sm mt-2">Ghost personas will appear here as they emerge</p>
          </div>
        ) : (
          <div ref={timelineRef} className="space-y-4">
            {/* Timeline line */}
            <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-slate-300 dark:bg-slate-600"></div>

            {filteredEvents.map((event, index) => (
              <motion.div
                key={event.id}
                layout
                initial={{ opacity: 0, x: -20 }}
                animate={{ 
                  opacity: 1, 
                  x: 0,
                  scale: playbackPosition === index && isPlaying ? 1.05 : 1
                }}
                transition={{ duration: 0.3 }}
                className={`
                  relative pl-16 pr-4 py-4 cursor-pointer transition-all duration-200
                  hover:bg-slate-50 dark:hover:bg-slate-800/50 rounded-lg
                  ${selectedEvent?.id === event.id ? 'bg-blue-50 dark:bg-blue-900/20 ring-1 ring-blue-200 dark:ring-blue-800' : ''}
                  ${playbackPosition === index && isPlaying ? 'ring-2 ring-purple-400 shadow-lg' : ''}
                `}
                onClick={() => handleEventClick(event)}
              >
                {/* Timeline marker */}
                <div className="absolute left-6 top-6">
                  <HoloObject
                    conceptId={`ghost-event-${event.id}`}
                    wavelength={event.wavelength}
                    intensity={0.8}
                    radius={12}
                  >
                    <div 
                      className="w-6 h-6 rounded-full border-2 border-white dark:border-slate-900 flex items-center justify-center text-xs"
                      style={{ backgroundColor: PERSONA_COLORS[event.persona] || '#64748b' }}
                    >
                      {getEventIcon(event.eventType)}
                    </div>
                  </HoloObject>
                </div>

                {/* Event content */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-slate-900 dark:text-slate-100">
                        {event.persona} {event.eventType}
                      </span>
                      <span 
                        className="px-2 py-0.5 rounded-full text-xs font-medium"
                        style={{ 
                          backgroundColor: `${PERSONA_COLORS[event.persona]}20`,
                          color: PERSONA_COLORS[event.persona]
                        }}
                      >
                        {event.eventType}
                      </span>
                    </div>
                    <span className="text-xs text-slate-500">
                      {formatEventTime(event.timestamp)}
                    </span>
                  </div>

                  {event.content?.message && (
                    <p className="text-sm text-slate-700 dark:text-slate-300 italic">
                      "{event.content.message.length > 100 
                        ? event.content.message.slice(0, 100) + '...'
                        : event.content.message}"
                    </p>
                  )}

                  <div className="text-xs text-slate-500 space-y-1">
                    <div>Trigger: {event.trigger.reason}</div>
                    {event.conceptIds && event.conceptIds.length > 0 && (
                      <div>Concepts: {event.conceptIds.join(', ')}</div>
                    )}
                    <div>Confidence: {Math.round(event.trigger.confidence * 100)}%</div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* Event Details Modal */}
      <AnimatePresence>
        {selectedEvent && (
          <motion.div
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedEvent(null)}
          >
            <motion.div
              className="bg-white dark:bg-slate-800 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto"
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <HoloObject
                      conceptId={`detail-${selectedEvent.id}`}
                      wavelength={selectedEvent.wavelength}
                      intensity={0.9}
                      radius={16}
                    >
                      <h3 className="text-lg font-semibold">
                        {selectedEvent.persona} Event Details
                      </h3>
                    </HoloObject>
                  </div>
                  <button
                    onClick={() => setSelectedEvent(null)}
                    className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                  >
                    ✕
                  </button>
                </div>

                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Event Information</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-slate-500">Type:</span> {selectedEvent.eventType}
                      </div>
                      <div>
                        <span className="text-slate-500">Time:</span> {selectedEvent.timestamp.toLocaleString()}
                      </div>
                      <div>
                        <span className="text-slate-500">Session:</span> {selectedEvent.sessionId}
                      </div>
                      <div>
                        <span className="text-slate-500">Wavelength:</span> {selectedEvent.wavelength}nm
                      </div>
                    </div>
                  </div>

                  {selectedEvent.content?.message && (
                    <div>
                      <h4 className="font-medium mb-2">Message</h4>
                      <p className="text-sm text-slate-700 dark:text-slate-300 italic bg-slate-50 dark:bg-slate-700/50 p-3 rounded">
                        "{selectedEvent.content.message}"
                      </p>
                    </div>
                  )}

                  <div>
                    <h4 className="font-medium mb-2">Trigger Context</h4>
                    <div className="text-sm space-y-1">
                      <div><span className="text-slate-500">Reason:</span> {selectedEvent.trigger.reason}</div>
                      <div><span className="text-slate-500">Confidence:</span> {Math.round(selectedEvent.trigger.confidence * 100)}%</div>
                      {selectedEvent.trigger.phaseSignature && (
                        <div><span className="text-slate-500">Phase:</span> {selectedEvent.trigger.phaseSignature}</div>
                      )}
                    </div>
                  </div>

                  {selectedEvent.conceptIds && selectedEvent.conceptIds.length > 0 && (
                    <div>
                      <h4 className="font-medium mb-2">Related Concepts</h4>
                      <div className="flex flex-wrap gap-2">
                        {selectedEvent.conceptIds.map(conceptId => (
                          <span
                            key={conceptId}
                            className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-400 rounded text-xs"
                          >
                            {conceptId}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default GhostChronicle;
export type { GhostEvent, MoodDataPoint };