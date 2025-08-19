import React, { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import { formatDistance } from 'date-fns';

/**
 * PsiarcTimelinePanel component for visualizing the psiarc timeline.
 * This component shows a list of concept graph changes over time and allows
 * scrubbing through the history to see the graph at different points in time.
 */
const PsiarcTimelinePanel = ({
  psiarc,
  onFrameSelect,
  currentFrameId,
  isPlaying,
  onPlayPause,
  playbackSpeed,
  onSpeedChange,
}) => {
  const [frames, setFrames] = useState([]);
  const [hoveredFrame, setHoveredFrame] = useState(null);
  const [filteredEventType, setFilteredEventType] = useState(null);
  const timelineRef = useRef(null);
  
  // Event type options for filtering
  const eventTypeOptions = [
    { id: null, label: 'All Events' },
    { id: 'ConceptCreated', label: 'Concept Created' },
    { id: 'ConceptUpdated', label: 'Concept Updated' },
    { id: 'ConceptLinked', label: 'Concept Linked' },
    { id: 'ConceptRemoved', label: 'Concept Removed' },
    { id: 'PlanRejected', label: 'Plan Rejected' },
    { id: 'PhaseUpdate', label: 'Phase Update' },
  ];
  
  // Load frames when psiarc changes
  useEffect(() => {
    if (!psiarc) {
      setFrames([]);
      return;
    }
    
    // In a real implementation, this would make an IPC call to
    // the Rust backend to load frames from the psiarc file
    const loadFrames = async () => {
      try {
        // This is a placeholder that would be replaced with actual loading logic
        // window.electron.invoke('psiarc:getFrames', psiarc)
        const loadedFrames = await simulateLoadFrames(psiarc);
        setFrames(loadedFrames);
      } catch (err) {
        console.error('Failed to load frames:', err);
      }
    };
    
    loadFrames();
  }, [psiarc]);
  
  // Scroll to current frame when it changes
  useEffect(() => {
    if (timelineRef.current && currentFrameId) {
      const frameEl = timelineRef.current.querySelector(`[data-frame-id="${currentFrameId}"]`);
      if (frameEl) {
        frameEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [currentFrameId]);
  
  // Filter frames based on selected event type
  const filteredFrames = filteredEventType 
    ? frames.filter(frame => frame.eventType === filteredEventType)
    : frames;
  
  // Calculate timeline markers (time indicators)
  const timeMarkers = calculateTimeMarkers(frames);
  
  // Handle timeline click to select a frame
  const handleTimelineClick = (frameId) => {
    onFrameSelect(frameId);
  };
  
  // Handle filter change
  const handleFilterChange = (e) => {
    setFilteredEventType(e.target.value === 'null' ? null : e.target.value);
  };
  
  // Handle play button click
  const handlePlayClick = () => {
    onPlayPause(!isPlaying);
  };
  
  // Handle speed change
  const handleSpeedChange = (e) => {
    onSpeedChange(parseFloat(e.target.value));
  };
  
  // Format timestamp for display
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };
  
  // Get relative time (e.g., "5 minutes ago")
  const getRelativeTime = (timestamp) => {
    return formatDistance(new Date(timestamp), new Date(), { addSuffix: true });
  };
  
  // Get color for event type
  const getEventColor = (eventType) => {
    switch (eventType) {
      case 'ConceptCreated':
        return 'bg-green-500';
      case 'ConceptUpdated':
        return 'bg-blue-500';
      case 'ConceptLinked':
        return 'bg-purple-500';
      case 'ConceptRemoved':
        return 'bg-red-500';
      case 'PlanRejected':
        return 'bg-orange-500';
      case 'PhaseUpdate':
        return 'bg-yellow-500';
      default:
        return 'bg-gray-500';
    }
  };
  
  // Get event icon
  const getEventIcon = (eventType) => {
    switch (eventType) {
      case 'ConceptCreated':
        return '➕';
      case 'ConceptUpdated':
        return '🔄';
      case 'ConceptLinked':
        return '🔗';
      case 'ConceptRemoved':
        return '❌';
      case 'PlanRejected':
        return '⛔';
      case 'PhaseUpdate':
        return '🔄';
      default:
        return '•';
    }
  };

  return (
    <div className="psiarc-timeline-panel h-full flex flex-col">
      {/* Header with controls */}
      <div className="p-4 bg-gray-800 border-b border-gray-700 flex items-center">
        <h2 className="text-lg font-semibold text-white mr-4">ψarc Timeline</h2>
        
        {/* File name */}
        {psiarc && (
          <span className="text-gray-300 mr-4">{getFileNameFromPath(psiarc)}</span>
        )}
        
        {/* Play controls */}
        <div className="flex items-center mr-4">
          <button 
            className="bg-blue-600 hover:bg-blue-700 text-white rounded-md px-3 py-1 mr-2"
            onClick={handlePlayClick}
          >
            {isPlaying ? '⏸ Pause' : '▶️ Play'}
          </button>
          
          <select 
            className="bg-gray-700 text-white rounded-md px-2 py-1"
            value={playbackSpeed}
            onChange={handleSpeedChange}
          >
            <option value="0.5">0.5x</option>
            <option value="1">1x</option>
            <option value="2">2x</option>
            <option value="5">5x</option>
          </select>
        </div>
        
        {/* Event type filter */}
        <div className="flex items-center">
          <label className="text-gray-300 mr-2">Filter:</label>
          <select 
            className="bg-gray-700 text-white rounded-md px-2 py-1"
            value={filteredEventType || 'null'}
            onChange={handleFilterChange}
          >
            {eventTypeOptions.map(option => (
              <option key={option.id} value={option.id || 'null'}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        
        {/* Info section */}
        <div className="ml-auto text-gray-300">
          <span>{frames.length} events</span>
          {currentFrameId && (
            <span className="ml-2">Viewing frame #{currentFrameId}</span>
          )}
        </div>
      </div>
      
      {/* Timeline view */}
      <div className="flex-1 overflow-y-auto" ref={timelineRef}>
        {/* Time markers */}
        <div className="sticky top-0 bg-gray-900 p-2 border-b border-gray-700 text-gray-300 text-sm flex">
          <div className="w-20">Time</div>
          <div className="w-24">Frame</div>
          <div className="flex-1">Event</div>
        </div>
        
        {/* Frame list */}
        {filteredFrames.length === 0 ? (
          <div className="p-4 text-gray-400 text-center">
            No events to display
          </div>
        ) : (
          <>
            {filteredFrames.map((frame, index) => (
              <React.Fragment key={frame.id}>
                {/* Add time marker if this is a new time segment */}
                {timeMarkers.includes(frame.timestamp) && (
                  <div className="bg-gray-800 text-gray-300 px-4 py-1 text-xs border-t border-b border-gray-700">
                    {formatTimestamp(frame.timestamp)} ({getRelativeTime(frame.timestamp)})
                  </div>
                )}
                
                {/* Frame item */}
                <div
                  data-frame-id={frame.id}
                  className={`flex p-2 border-b border-gray-800 hover:bg-gray-800 cursor-pointer ${
                    currentFrameId === frame.id ? 'bg-blue-900' : ''
                  }`}
                  onClick={() => handleTimelineClick(frame.id)}
                  onMouseEnter={() => setHoveredFrame(frame)}
                  onMouseLeave={() => setHoveredFrame(null)}
                >
                  {/* Timestamp */}
                  <div className="w-20 text-gray-400 text-sm">
                    {formatTimestamp(frame.timestamp).split(' ')[0]}
                  </div>
                  
                  {/* Frame ID */}
                  <div className="w-24 text-gray-400 text-sm">
                    #{frame.id}
                  </div>
                  
                  {/* Event */}
                  <div className="flex-1 flex items-center">
                    {/* Event type indicator */}
                    <span className={`w-4 h-4 rounded-full mr-2 flex items-center justify-center ${getEventColor(frame.eventType)}`} title={frame.eventType}>
                      {getEventIcon(frame.eventType)}
                    </span>
                    
                    {/* Event description */}
                    <span className="text-white">
                      {frame.description}
                    </span>
                  </div>
                </div>
              </React.Fragment>
            ))}
          </>
        )}
      </div>
      
      {/* Hover detail panel */}
      {hoveredFrame && (
        <div className="bg-gray-800 p-4 border-t border-gray-700">
          <h3 className="text-white font-semibold mb-2">
            {hoveredFrame.eventType} (Frame #{hoveredFrame.id})
          </h3>
          <div className="text-gray-300">
            <div><span className="text-gray-400">Time:</span> {formatTimestamp(hoveredFrame.timestamp)}</div>
            <div><span className="text-gray-400">Source:</span> {hoveredFrame.source}</div>
            {hoveredFrame.data && (
              <div className="mt-2">
                <span className="text-gray-400">Data:</span>
                <pre className="bg-gray-900 p-2 rounded mt-1 text-sm overflow-x-auto">
                  {JSON.stringify(hoveredFrame.data, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Prop types
PsiarcTimelinePanel.propTypes = {
  /** Path to the psiarc file */
  psiarc: PropTypes.string,
  /** Callback when a frame is selected */
  onFrameSelect: PropTypes.func.isRequired,
  /** Currently selected frame ID */
  currentFrameId: PropTypes.number,
  /** Whether the timeline is playing */
  isPlaying: PropTypes.bool,
  /** Callback when play/pause is toggled */
  onPlayPause: PropTypes.func.isRequired,
  /** Playback speed (0.5, 1, 2, etc.) */
  playbackSpeed: PropTypes.number,
  /** Callback when playback speed is changed */
  onSpeedChange: PropTypes.func.isRequired,
};

PsiarcTimelinePanel.defaultProps = {
  isPlaying: false,
  playbackSpeed: 1,
};

// Helper functions
function getFileNameFromPath(path) {
  return path ? path.split('/').pop() : '';
}

function calculateTimeMarkers(frames) {
  if (!frames.length) return [];
  
  // Group frames by minute for the markers
  const markers = [];
  let lastMinute = null;
  
  frames.forEach(frame => {
    const date = new Date(frame.timestamp);
    const minute = Math.floor(date.getTime() / (1000 * 60));
    
    if (minute !== lastMinute) {
      markers.push(frame.timestamp);
      lastMinute = minute;
    }
  });
  
  return markers;
}

// Simulation helper - replace with actual implementation
async function simulateLoadFrames(psiarc) {
  // This would be replaced with actual loading from the backend
  return [
    {
      id: 1,
      timestamp: Date.now() - 10000,
      eventType: 'ConceptCreated',
      description: 'Created concept "Document"',
      source: 'IngestAgent',
      data: { conceptId: 'doc1', type: 'Document' }
    },
    {
      id: 2,
      timestamp: Date.now() - 9000,
      eventType: 'ConceptCreated',
      description: 'Created concept "Section"',
      source: 'IngestAgent',
      data: { conceptId: 'section1', type: 'Section' }
    },
    {
      id: 3,
      timestamp: Date.now() - 8000,
      eventType: 'ConceptLinked',
      description: 'Linked "Document" to "Section"',
      source: 'IngestAgent',
      data: { sourceId: 'doc1', targetId: 'section1', relationType: 'contains' }
    },
    {
      id: 4,
      timestamp: Date.now() - 7000,
      eventType: 'PhaseUpdate',
      description: 'Phase update for "Document"',
      source: 'OscillatorAgent',
      data: { conceptId: 'doc1', phase: 0.25 }
    },
    {
      id: 5,
      timestamp: Date.now() - 6000,
      eventType: 'PlanRejected',
      description: 'Plan rejected: Instability risk detected',
      source: 'PlannerAgent',
      data: { planId: 'plan1', reason: 'Instability risk: V̇ = 0.349 > threshold 0.200' }
    },
  ];
}

export default PsiarcTimelinePanel;
