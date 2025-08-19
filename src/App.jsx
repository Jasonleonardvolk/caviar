import React, { useState, useEffect, useRef } from 'react';
import { Controlled as CodeMirrorEditor } from 'react-codemirror2';
import QuickActionsBar from './components/QuickActionsBar/QuickActionsBar';
import QuickActionsPanel from './components/QuickActionsPanel/QuickActionsPanel';
import agentSuggestionsService from './services/agentSuggestionsService';
import 'codemirror/lib/codemirror.css';
import 'codemirror/theme/material.css';
import 'codemirror/mode/javascript/javascript';
import 'codemirror/mode/jsx/jsx';
import './App.css';

function App() {
  const [agentSuggestions, setAgentSuggestions] = useState([]);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [editorValue, setEditorValue] = useState('// Write your code here\n\nfunction helloWorld() {\n  console.log("Hello, world!");\n}\n');
  const editorRef = useRef(null);
  const wsConnectionRef = useRef(null);
  
  useEffect(() => {
    // Initial fetch
    const fetchInitialSuggestions = async () => {
      try {
        const suggestions = await agentSuggestionsService.fetchSuggestions();
        setAgentSuggestions(suggestions);
      } catch (error) {
        console.error('Error fetching initial suggestions:', error);
      }
    };
    
    fetchInitialSuggestions();
    
    // Connect to WebSocket for real-time updates
    wsConnectionRef.current = agentSuggestionsService.connectWebSocket(
      (newSuggestions) => setAgentSuggestions(newSuggestions)
    );
    
    // Cleanup WebSocket connection on component unmount
    return () => {
      if (wsConnectionRef.current) {
        wsConnectionRef.current.disconnect();
      }
    };
  }, []);
  
  const handleApplySuggestion = async (suggestion) => {
    try {
      // If suggestion has its own handler, use that
      if (suggestion.onApply) {
        suggestion.onApply();
        // Remove the suggestion from the list after applying
        setAgentSuggestions(prev => prev.filter(s => s.id !== suggestion.id));
        return;
      }
      
      // Apply suggestion to the editor if it has diff information
      if (suggestion.diff && editorRef.current) {
        const editor = editorRef.current.editor;
        
        // If we have specific range information
        if (suggestion.rangeStart && suggestion.rangeEnd) {
          editor.replaceRange(
            suggestion.diff.new,
            suggestion.rangeStart,
            suggestion.rangeEnd
          );
        } else {
          // Otherwise, just do a simple text replacement
          // This is less precise but works for simple cases
          const content = editor.getValue();
          const updatedContent = content.replace(suggestion.diff.old, suggestion.diff.new);
          if (content !== updatedContent) {
            editor.setValue(updatedContent);
          }
        }
      }
      
      // Call the API
      await agentSuggestionsService.applySuggestion(suggestion.id);
      
      // Remove the suggestion from the list after applying
      setAgentSuggestions(prev => prev.filter(s => s.id !== suggestion.id));
    } catch (error) {
      console.error('Error applying suggestion:', error);
      // Handle error (e.g., show a notification)
    }
  };
  
  const handleDismissSuggestion = async (suggestion) => {
    try {
      await agentSuggestionsService.dismissSuggestion(suggestion.id);
      // Remove the suggestion from the list
      setAgentSuggestions(prev => prev.filter(s => s.id !== suggestion.id));
    } catch (error) {
      console.error('Error dismissing suggestion:', error);
      // Handle error (e.g., show a notification)
    }
  };
  
  const openPanel = () => setIsPanelOpen(true);
  const closePanel = () => setIsPanelOpen(false);

  return (
    <div className="alan-ide-root">
      <div className="main-editor-area">
        <div className="code-editor">
          <CodeMirrorEditor
            ref={editorRef}
            value={editorValue}
            options={{
              mode: 'jsx',
              theme: 'material',
              lineNumbers: true,
              lineWrapping: true,
              extraKeys: { "Ctrl-Space": "autocomplete" }
            }}
            onBeforeChange={(editor, data, value) => {
              setEditorValue(value);
            }}
          />
        </div>
      </div>
      
      {/* Quick Actions Bar */}
      <QuickActionsBar 
        suggestions={agentSuggestions}
        onOpenPanel={openPanel}
        onApply={handleApplySuggestion}
        onDismiss={handleDismissSuggestion}
      />
      
      {/* Quick Actions Panel (expanded view) */}
      <QuickActionsPanel
        isOpen={isPanelOpen}
        onClose={closePanel}
        suggestions={agentSuggestions}
        onApply={handleApplySuggestion}
        onDismiss={handleDismissSuggestion}
      />
    </div>
  );
}

export default App;
