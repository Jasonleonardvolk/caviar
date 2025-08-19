import React, { useState } from "react";
import QuickActionItem from "./QuickActionItem";
import "./QuickActionsBar.css";

// Helper to render icon from string or JSX
const renderIcon = (icon) => {
  if (typeof icon === 'string') {
    return <span role="img" aria-label="Agent Icon">{icon}</span>;
  }
  return icon;
};

export default function QuickActionsBar({ 
  suggestions = [], 
  onOpenPanel,
  onApply: applyHandler,
  onDismiss: dismissHandler
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [hoveredSuggestion, setHoveredSuggestion] = useState(null);

  const handleApply = (suggestion) => {
    // Use the suggestion's onApply if available, otherwise use the passed handler
    if (suggestion.onApply) {
      suggestion.onApply();
    } else if (applyHandler) {
      applyHandler(suggestion);
    }
  };

  const handleDismiss = (suggestion) => {
    if (dismissHandler) {
      dismissHandler(suggestion);
    }
  };

  // Simplified view when collapsed - show as a horizontal bar
  if (!isExpanded) {
    return (
      <div className="quick-actions-bar">
        <div className="quick-actions-header">
          <div className="quick-actions-title" onClick={() => setIsExpanded(true)}>
            <span className="icon">💡</span>
            <span>Agent Suggestions ({suggestions.length})</span>
          </div>
          <div className="header-actions">
            <button className="view-all-button" onClick={onOpenPanel || (() => alert("Show all suggestions panel"))}>
              View All
            </button>
            <button className="expand-button" onClick={() => setIsExpanded(true)}>
              ▲
            </button>
          </div>
        </div>
        
        <div className="quick-actions-horizontal" style={{ display: "flex", paddingLeft: "16px", paddingBottom: "8px" }}>
          {suggestions.map((suggestion) => (
            <QuickActionItem
              key={suggestion.id}
              {...suggestion}
              onApply={() => handleApply(suggestion)}
            />
          ))}
        </div>
      </div>
    );
  }
  
  // Expanded view with details
  return (
    <div className="quick-actions-bar expanded">
      <div className="quick-actions-header">
        <div className="quick-actions-title">
          <span className="icon">💡</span>
          <span>Agent Suggestions ({suggestions.length})</span>
        </div>
        <div className="header-actions">
          <button className="view-all-button" onClick={onOpenPanel || (() => alert("Show all suggestions panel"))}>
            View All
          </button>
          <button className="expand-button" onClick={() => setIsExpanded(false)}>
            ▼
          </button>
        </div>
      </div>
      
      <div className="quick-actions-suggestions">
        {suggestions.map((suggestion) => (
          <div 
            key={suggestion.id}
            className="suggestion-item"
            style={{ borderLeftColor: suggestion.color }}
            onMouseEnter={() => setHoveredSuggestion(suggestion)}
            onMouseLeave={() => setHoveredSuggestion(null)}
          >
            <div className="suggestion-icon" style={{ backgroundColor: suggestion.color }}>
              {renderIcon(suggestion.icon)}
            </div>
            <div className="suggestion-content">
              <div className="suggestion-label">{suggestion.label}</div>
              {hoveredSuggestion === suggestion && (
                <div className="suggestion-explanation">{suggestion.explanation}</div>
              )}
            </div>
            <div className="suggestion-actions">
              <button 
                className="apply-button"
                onClick={() => handleApply(suggestion)}
              >
                Apply
              </button>
              <button 
                className="dismiss-button"
                onClick={() => handleDismiss(suggestion)}
              >
                Dismiss
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
