// QuickActionsPanel.jsx
import React from 'react';
import './QuickActionsPanel.css';

// Import a code diff library (you'll need to install this)
// npm install react-diff-viewer
// import ReactDiffViewer from 'react-diff-viewer';

const QuickActionsPanel = ({ 
  isOpen, 
  onClose, 
  suggestions, 
  onApply, 
  onDismiss 
}) => {
  if (!isOpen) return null;

  // Group suggestions by persona
  const groupedSuggestions = suggestions.reduce((acc, suggestion) => {
    if (!acc[suggestion.persona]) {
      acc[suggestion.persona] = [];
    }
    acc[suggestion.persona].push(suggestion);
    return acc;
  }, {});

  return (
    <div className="quick-actions-panel-overlay">
      <div className="quick-actions-panel">
        <div className="panel-header">
          <h2>Agent Suggestions</h2>
          <button className="close-button" onClick={onClose}>✕</button>
        </div>
        
        <div className="panel-content">
          {Object.entries(groupedSuggestions).map(([persona, personaSuggestions]) => (
            <div key={persona} className="persona-group">
              <h3 className="persona-title">{persona}</h3>
              
              {personaSuggestions.map(suggestion => (
                <div 
                  key={suggestion.id} 
                  className="panel-suggestion"
                  style={{ borderLeftColor: suggestion.color }}
                >
                  <div className="panel-suggestion-header">
                    <div className="panel-suggestion-icon" style={{ backgroundColor: suggestion.color }}>
                      {suggestion.icon}
                    </div>
                    <div className="panel-suggestion-title">{suggestion.label}</div>
                    <div className="panel-suggestion-actions">
                      <button 
                        className="panel-apply-button"
                        onClick={() => onApply(suggestion)}
                      >
                        Apply
                      </button>
                      <button 
                        className="panel-dismiss-button"
                        onClick={() => onDismiss(suggestion)}
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                  
                  <div className="panel-suggestion-explanation">
                    {suggestion.explanation}
                  </div>
                  
                  <div className="panel-suggestion-diff">
                    {/* Uncomment when react-diff-viewer is installed */}
                    {/* <ReactDiffViewer
                      oldValue={suggestion.diff.old}
                      newValue={suggestion.diff.new}
                      splitView={true}
                      useDarkTheme={true}
                      leftTitle="Current Code"
                      rightTitle="Suggested Code"
                    /> */}
                    
                    {/* Temporary fallback until diff viewer is installed */}
                    <div className="diff-fallback">
                      <div className="diff-section">
                        <h4>Current Code:</h4>
                        <pre>{suggestion.diff.old}</pre>
                      </div>
                      <div className="diff-section">
                        <h4>Suggested Code:</h4>
                        <pre>{suggestion.diff.new}</pre>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default QuickActionsPanel;