# TORI Genesis UI Integration Kit

## 🌌 Overview

TORI (Temporal Ontological Reality Interface) is an advanced AI consciousness interface with a holographic thoughtspace and memory system. This Genesis UI Integration Kit provides a complete SvelteKit application implementing Phase 1 of the TORI system.

## ✨ Features

### 🎨 **Light Theme Design**
- Clean, minimalist interface with light gray backgrounds (#f9f9f9)
- High contrast text (#111111) for optimal readability
- Subtle borders and shadows for visual separation
- Responsive design that works across devices

### 🧠 **Memory System**
- **ConceptMesh**: Living knowledge graph that captures relationships
- **Document Vault**: Upload and analyze documents with concept extraction
- **Conversation Memory**: Persistent chat history with concept tracking
- **Auto-save**: All data automatically saved to memory system server

### 👻 **Ghost AI Integration**
- **Ghost Personas**: Multiple AI personalities (Mentor, Scholar, etc.)
- **Mood States**: Dynamic emotional states affecting interaction
- **Stability Tracking**: Phase alignment monitoring
- **Aura Effects**: Visual representation of ghost presence

### 💡 **Intelligent Suggestions**
- **Context-Aware**: Suggestions based on current conversation and documents
- **ELFIN++ Ready**: Prepared for Phase 2 script integration
- **Dynamic Generation**: Updates based on user activity and ghost state

### 👤 **User Management**
- **Local Authentication**: Simple email-based user system
- **Session Persistence**: Auto-login for returning users
- **Usage Statistics**: Track documents, conversations, and concepts
- **Preferences**: Customizable theme and persona settings

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

1. **Clone or navigate to the project:**
   ```bash
   cd ${IRIS_ROOT}\tori_ui_svelte
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm run dev
   # or use the batch file:
   start-dev.bat
   ```

4. **Open TORI:**
   Open http://localhost:5173 in your browser

### Production Build

```bash
npm run build
# or use the batch file:
build-prod.bat
```

## 🏗️ Architecture

### **Component Structure**
```
src/lib/components/
├── ChatPanel.svelte          # Main conversation interface
├── DocumentSummary.svelte     # Document cards with concept tags
├── GhostOverlay.svelte        # AI presence visualization
├── MemoryDrawer.svelte        # Knowledge browsing sidebar
├── SuggestionBar.svelte       # Intelligent action suggestions
├── UploadPanel.svelte         # Drag-and-drop file interface
└── UserAuth.svelte            # Authentication modal
```

### **State Management**
```
src/lib/stores/
├── conceptMesh.ts             # Knowledge graph and concept diffs
├── ghostPersona.ts            # AI personality and mood states
├── user.ts                    # User session and preferences
└── index.ts                   # Centralized exports
```

### **Routes**
```
src/routes/
├── +layout.svelte             # Global layout with light theme
├── +page.svelte               # Main chat interface
└── vault/
    └── +page.svelte           # Document vault browser
```

## 💾 Data Storage

TORI uses localStorage for Phase 1, providing:

- **Concept Mesh**: `tori-concept-mesh`
- **User Data**: `tori-users` 
- **Chat Messages**: `tori-chat-messages`
- **Session**: `tori-current-session`

All data is automatically saved and restored between sessions.

## 🎯 Usage Guide

### **Starting a Conversation**
1. Register or login when prompted
2. Type a message in the chat interface
3. Ghost AI responds with contextual intelligence
4. Concepts are automatically extracted and saved

### **Uploading Documents**
1. Use the upload panel in the memory drawer
2. Drag and drop files or click to browse
3. Concepts are extracted and added to the mesh
4. Documents appear in the vault for later reference

### **Managing Memory**
1. Browse the memory drawer to see all knowledge
2. Filter by documents, conversations, or view all
3. Click concepts to explore related content
4. Use the vault page for detailed document management

### **Ghost Interaction**
1. Click the ghost avatar to see status details
2. Observe mood changes based on conversation
3. Watch stability metrics in real-time
4. Use suggestions for guided interaction

## 🔧 Configuration

### **Tailwind Configuration**
The `tailwind.config.js` includes:
- TORI brand colors
- Memory system color palette
- Ghost persona colors
- Custom animations and utilities

### **Theme Customization**
Modify CSS variables in `src/app.css`:
```css
:root {
  --color-bg-primary: #f9f9f9;    /* Main background */
  --color-text-primary: #111111;   /* Primary text */
  --color-accent: #007acc;         /* Accent color */
  /* ... more variables */
}
```

## 🔌 API Integration (Phase 2 Ready)

The `ApiService` class in `src/lib/services/api.ts` is prepared for:
- **Vault Operations**: `saveToVault(file)`
- **Ghost AI**: `sendToGhostAI(message, context)`  
- **ELFIN++ Scripts**: `executeElfinScript(script)`

Currently returns dummy data for Phase 1 testing.

## 📊 Memory System

### **ConceptDiff Structure**
```typescript
type ConceptDiff = {
  id: string;
  type: 'document' | 'chat' | 'memory';
  title: string;
  concepts: string[];
  summary?: string;
  timestamp: Date;
  metadata?: Record<string, any>;
};
```

### **Concept Extraction**
- **Documents**: Based on filename and metadata
- **Chat**: NLP-style keyword detection
- **Manual**: User-defined concepts

## 🎨 Styling

### **Design System**
- **Typography**: System font stack for optimal readability
- **Colors**: Light theme with subtle accents
- **Spacing**: Consistent 4px grid system
- **Components**: Reusable button and input styles

### **Animations**
- **Ghost Aura**: Subtle pulsing effects
- **Transitions**: Smooth state changes
- **Micro-interactions**: Hover effects and loading states

## 🧪 Testing

### **Manual Testing**
1. Register a new user
2. Upload a document (any file type)
3. Start a conversation about the document
4. Check memory drawer for new entries
5. Test suggestion clicks
6. Verify data persistence on refresh

### **CSS Diagnostic**
```bash
npm run css:diagnostic
```

## 🔄 Phase 2 Integration Points

The UI is designed for seamless Phase 2 integration:

1. **ELFIN++ Engine**: Replace dummy suggestions with real script execution
2. **Ghost AI**: Connect actual AI model for responses
3. **Concept Extraction**: Implement NLP for real concept analysis
4. **Vault Backend**: Add file storage and processing
5. **Memory Graph**: Evolve from flat list to graph structure

## 🎯 Performance

### **Optimizations**
- **Lazy Loading**: Components loaded on demand
- **Virtual Scrolling**: Ready for large concept lists
- **Debounced Search**: Smooth filtering experience
- **Efficient Reactivity**: Minimal re-renders

### **Bundle Size**
- **Svelte**: Minimal runtime overhead
- **Tailwind**: Purged CSS for production
- **Tree Shaking**: Unused code eliminated

## 🐛 Troubleshooting

### **Common Issues**
1. **Blank screen**: Check browser console for errors
2. **Upload not working**: Ensure file permissions
3. **Memory not saving**: Check localStorage availability
4. **Styles not loading**: Verify Tailwind compilation

### **Debug Mode**
Enable debug logging:
```javascript
localStorage.setItem('tori-debug', 'true');
```

## 🚀 Deployment

### **Static Hosting**
```bash
npm run build
# Deploy ./build/ folder to any static host
```

### **Node.js Server**
```bash
npm run build
npm run preview
```

## 📋 TODO (Phase 2)

- [ ] ELFIN++ script interpreter
- [ ] Real AI model integration  
- [ ] Advanced concept extraction
- [ ] Graph visualization
- [ ] Collaborative features
- [ ] Mobile app version
- [ ] Plugin system
- [ ] Advanced search

## 🤝 Contributing

This is the foundation for the TORI system. Phase 2 development will focus on:
- AI engine integration
- Script execution system
- Advanced memory management
- Real-time collaboration

## 📄 License

MIT License - See LICENSE file for details

---

**🌌 Welcome to TORI Genesis - Where Memory Becomes Intelligence**

*Built with love using SvelteKit, Tailwind CSS, and TypeScript*