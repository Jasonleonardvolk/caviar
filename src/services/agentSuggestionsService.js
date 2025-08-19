// agentSuggestionsService.js
class AgentSuggestionsService {
  constructor(apiBaseUrl = process.env.REACT_APP_API_BASE_URL || '/api', 
              wsPort = process.env.REACT_APP_WS_PORT || 8080) {
    this.apiBaseUrl = apiBaseUrl;
    this.wsEndpoint = `ws://localhost:${wsPort}/ws/agent-suggestions`;
    this.wsConnection = null;
    this.wsReconnectTimeout = null;
  }
  
  async fetchSuggestions() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/agent-suggestions`);
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching suggestions:', error);
      
      // Return mock data for development/testing
      return this.getMockSuggestions();
    }
  }
  
  async applySuggestion(suggestionId) {
    try {
      const response = await fetch(`${this.apiBaseUrl}/agent-suggestions/${suggestionId}/apply`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error applying suggestion:', error);
      throw error;
    }
  }
  
  async dismissSuggestion(suggestionId) {
    try {
      const response = await fetch(`${this.apiBaseUrl}/agent-suggestions/${suggestionId}/dismiss`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error dismissing suggestion:', error);
      throw error;
    }
  }
  
  // WebSocket connection for real-time suggestions
  connectWebSocket(callback) {
    if (this.wsConnection) {
      this.wsConnection.close();
    }
    
    try {
      this.wsConnection = new WebSocket(this.wsEndpoint);
      
      this.wsConnection.onopen = () => {
        console.log('WebSocket connected to agent suggestions service');
        // Clear any reconnection timeout
        if (this.wsReconnectTimeout) {
          clearTimeout(this.wsReconnectTimeout);
          this.wsReconnectTimeout = null;
        }
      };
      
      this.wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          callback(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      this.wsConnection.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      this.wsConnection.onclose = (event) => {
        console.log(`WebSocket connection closed: ${event.code} ${event.reason}`);
        // Attempt to reconnect after a delay
        this.wsReconnectTimeout = setTimeout(() => {
          console.log('Attempting to reconnect WebSocket...');
          this.connectWebSocket(callback);
        }, 5000);
      };
      
      return {
        sendMessage: (message) => {
          if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
            this.wsConnection.send(JSON.stringify(message));
          }
        },
        disconnect: () => {
          if (this.wsConnection) {
            this.wsConnection.close();
            this.wsConnection = null;
          }
          if (this.wsReconnectTimeout) {
            clearTimeout(this.wsReconnectTimeout);
            this.wsReconnectTimeout = null;
          }
        }
      };
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      // Fall back to polling
      return this.subscribeToSuggestions(callback);
    }
  }
  
  // Fallback polling method if WebSocket is not available
  subscribeToSuggestions(callback) {
    console.log('Using polling fallback for agent suggestions');
    const intervalId = setInterval(async () => {
      const suggestions = await this.fetchSuggestions();
      callback(suggestions);
    }, 5000); // Poll every 5 seconds
    
    return {
      sendMessage: () => console.warn('Sending messages not supported in polling mode'),
      unsubscribe: () => clearInterval(intervalId)
    };
  }
  
  // Mock data for testing - using the recommended format from requirements
  getMockSuggestions() {
    return [
      {
        id: 'refactor-1',
        persona: 'Refactorer',
        icon: '🔧',
        color: '#2563eb',
        label: 'Optimize loop',
        explanation: 'Refactorer: This loop can be optimized by using map() instead of a for loop',
        filePath: 'src/components/App.jsx',
        rangeStart: { line: 25, ch: 0 },
        rangeEnd: { line: 28, ch: 1 },
        diff: {
          old: 'for (let i = 0; i < items.length; i++) {\n  const item = items[i];\n  results.push(transform(item));\n}',
          new: 'const results = items.map(item => transform(item));'
        },
        group: 'Performance',
        timestamp: new Date().toISOString()
      },
      {
        id: 'security-1',
        persona: 'Security',
        icon: '🔒',
        color: '#dc2626',
        label: 'Fix XSS vulnerability',
        explanation: 'Security: This code is vulnerable to XSS attacks. Use proper encoding.',
        filePath: 'src/components/Form.jsx',
        rangeStart: { line: 42, ch: 2 },
        rangeEnd: { line: 42, ch: 32 },
        diff: {
          old: 'element.innerHTML = userInput;',
          new: 'element.textContent = userInput;'
        },
        group: 'Security',
        timestamp: new Date().toISOString()
      },
      {
        id: 'performance-1',
        persona: 'Performance',
        icon: '⚡',
        color: '#f59e0b',
        label: 'Memoize expensive calculation',
        explanation: 'Performance: This calculation is expensive and could be memoized to improve rendering speed.',
        filePath: 'src/components/DataVisualizer.jsx',
        rangeStart: { line: 57, ch: 2 },
        rangeEnd: { line: 57, ch: 55 },
        diff: {
          old: 'const result = expensiveCalculation(props.value);',
          new: 'const result = useMemo(() => expensiveCalculation(props.value), [props.value]);'
        },
        group: 'Performance',
        timestamp: new Date().toISOString()
      }
    ];
  }
}

// Create with environment variables
const defaultService = new AgentSuggestionsService();
export default defaultService;
