import React, { useState } from 'react';
import './App.css';

function App() {
  const [suggestions, setSuggestions] = useState([
    {
      id: 'refactor-1',
      persona: 'Refactorer',
      icon: '🔧',
      color: '#00FFCC',
      label: 'Optimize loop',
      explanation: 'This loop can be optimized by using map() instead of a for loop',
      filePath: 'src/components/App.jsx',
      rangeStart: { line: 25, ch: 0 },
      rangeEnd: { line: 28, ch: 1 },
      diff: {
        old: 'for (let i = 0; i < items.length; i++) {\n  const item = items[i];\n  results.push(transform(item));\n}',
        new: 'const results = items.map(item => transform(item));'
      },
      group: 'Performance',
    },
    {
      id: 'security-1',
      persona: 'Security',
      icon: '🔒',
      color: '#FF007F',
      label: 'Fix XSS vulnerability',
      explanation: 'This code is vulnerable to XSS attacks. Use proper encoding.',
      filePath: 'src/components/Form.jsx',
      rangeStart: { line: 42, ch: 2 },
      rangeEnd: { line: 42, ch: 32 },
      diff: {
        old: 'element.innerHTML = userInput;',
        new: 'element.textContent = userInput;'
      },
      group: 'Security',
    },
    {
      id: 'performance-1',
      persona: 'Performance',
      icon: '⚡',
      color: '#FFD700',
      label: 'Memoize expensive calculation',
      explanation: 'This calculation is expensive and could be memoized to improve rendering speed.',
      filePath: 'src/components/DataVisualizer.jsx',
      rangeStart: { line: 57, ch: 2 },
      rangeEnd: { line: 57, ch: 55 },
      diff: {
        old: 'const result = expensiveCalculation(props.value);',
        new: 'const result = useMemo(() => expensiveCalculation(props.value), [props.value]);'
      },
      group: 'Performance',
    }
  ]);

  const [activeTab, setActiveTab] = useState('suggestions');

  const handleApply = (id) => {
    setSuggestions(suggestions.filter(suggestion => suggestion.id !== id));
  };

  const handleDismiss = (id) => {
    setSuggestions(suggestions.filter(suggestion => suggestion.id !== id));
  };

  return (
    <div className="alan-ide-layout">
      <header className="alan-header">
        <div className="alan-logo">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 7V17L12 22L22 17V7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
            <path d="M12 22V12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M12 12L22 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M12 12L2 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          ALAN <span className="alan-title">IDE | Phase 3</span>
        </div>
        
        <div className="persona-selector">
          <button className="persona-current-btn">
            <div className="persona-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24">
                <path d="M15 13.5V8h-2v5.5l-1.5-1.5L10 13.5 12 15.5l2 2 2-2 2-2-1.5-1.5L15 13.5zM12 3C6.5 3 2 7.5 2 13c0 5.5 4.5 10 10 10s10-4.5 10-10c0-5.5-4.5-10-10-10zm0 18c-4.4 0-8-3.6-8-8s3.6-8 8-8 8 3.6 8 8-3.6 8-8 8z"/>
              </svg>
            </div>
            <span>Concept Architect</span>
          </button>
        </div>
      </header>
      
      <main className="concept-field-canvas">
        <div className="canvas-overlay"></div>
        <div className="canvas-placeholder">
          <h2>Concept Field Visualization</h2>
          <p>This area displays the interactive concept field canvas where nodes represent code concepts and edges show relationships between them. The visual overlays display phase dynamics, coupling strengths, and Koopman operator projections.</p>
        </div>
      </main>
      
      <aside className="panel-dock">
        <div className="panel-tabs">
          <div 
            className={`panel-tab ${activeTab === 'debug' ? 'active' : ''}`}
            onClick={() => setActiveTab('debug')}
          >
            Debug
          </div>
          <div 
            className={`panel-tab ${activeTab === 'doc' ? 'active' : ''}`}
            onClick={() => setActiveTab('doc')}
          >
            Doc
          </div>
          <div 
            className={`panel-tab ${activeTab === 'suggestions' ? 'active' : ''}`}
            onClick={() => setActiveTab('suggestions')}
          >
            Suggestions
          </div>
        </div>
        
        <div className="panel-content">
          {activeTab === 'suggestions' && (
            <div className="suggestion-list">
              {suggestions.map(suggestion => (
                <div className="suggestion-card" key={suggestion.id}>
                  <div className="suggestion-header">
                    <div className="persona-icon" style={{ color: suggestion.color }}>
                      {suggestion.icon}
                    </div>
                    <div className="suggestion-label">{suggestion.label}</div>
                  </div>
                  <div className="suggestion-body">
                    <div className="suggestion-explanation">
                      {suggestion.explanation}
                    </div>
                    <div className="suggestion-code">
                      <code className="code-old">{suggestion.diff.old}</code>
                      <code className="code-new">{suggestion.diff.new}</code>
                    </div>
                    <div className="suggestion-actions">
                      <button 
                        className="alan-button" 
                        onClick={() => handleDismiss(suggestion.id)}
                      >
                        Dismiss
                      </button>
                      <button 
                        className="alan-button primary" 
                        onClick={() => handleApply(suggestion.id)}
                      >
                        Apply
                      </button>
                    </div>
                  </div>
                </div>
              ))}

              {suggestions.length === 0 && (
                <div className="suggestion-empty">
                  <p>No agent suggestions available at this time.</p>
                  <p>Try opening some source code files or making edits to generate suggestions.</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'debug' && (
            <div className="debug-panel">
              <h3>Debug Agent</h3>
              <p>The Debug Agent will analyze your code for potential issues and anomalies.</p>
              <p>Start coding to receive debug suggestions.</p>
            </div>
          )}

          {activeTab === 'doc' && (
            <div className="doc-panel">
              <h3>Documentation Agent</h3>
              <p>The Documentation Agent will help you maintain high-quality code documentation.</p>
              <p>Open source files to receive documentation suggestions.</p>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

export default App;
