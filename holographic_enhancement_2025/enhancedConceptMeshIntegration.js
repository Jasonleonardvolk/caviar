/**
 * Enhanced Concept Mesh Integration with Complete TODO Implementation
 * Handles all concept operations including deletion, relation updates, and graceful degradation
 */

import { holographicSystem } from './unifiedHolographicSystem';

export class EnhancedConceptMeshIntegration {
  constructor() {
    this.ws = null;
    this.reconnectTimeout = null;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.isConnected = false;
    this.messageQueue = [];
    this.conceptHistory = [];
    this.maxHistorySize = 100;
    this.offlineMode = false;
    
    // Concept and relation caches for offline operation
    this.conceptCache = new Map();
    this.relationCache = new Map();
    
    // Deletion tracking
    this.deletedConcepts = new Set();
    this.deletedRelations = new Set();
  }
  
  connect(url = 'ws://localhost:8766/concepts') {
    console.log('🔌 Connecting to Concept Mesh...');
    
    try {
      this.ws = new WebSocket(url);
      
      this.ws.onopen = () => {
        console.log('✅ Connected to Concept Mesh');
        this.isConnected = true;
        this.offlineMode = false;
        this.reconnectDelay = 1000; // Reset delay
        
        // Send any queued messages
        this.flushMessageQueue();
        
        // Request initial scene
        this.send({ type: 'get_scene' });
      };
      
      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      };
      
      this.ws.onerror = (error) => {
        console.error('Concept Mesh error:', error);
        this.handleConnectionError();
      };
      
      this.ws.onclose = () => {
        console.log('Disconnected from Concept Mesh');
        this.isConnected = false;
        this.scheduleReconnect();
      };
      
    } catch (error) {
      console.error('Failed to connect to Concept Mesh:', error);
      this.handleConnectionError();
    }
  }
  
  handleConnectionError() {
    console.warn('⚠️ Entering offline mode - using cached concepts');
    this.offlineMode = true;
    this.isConnected = false;
    
    // Load cached scene if available
    this.loadCachedScene();
    
    // Notify user
    if (holographicSystem) {
      holographicSystem.showNotification('Concept Mesh offline - operating with cached data', 'warning');
    }
  }
  
  handleMessage(data) {
    // Store for potential rollback
    this.addToHistory({
      type: 'message_received',
      timestamp: Date.now(),
      data: JSON.parse(JSON.stringify(data))
    });
    
    switch (data.type) {
      case 'holographic_scene':
        console.log(`Loading scene with ${data.scene.total_concepts} concepts`);
        this.loadScene(data.scene);
        this.cacheScene(data.scene);
        break;
        
      case 'mesh_update':
        this.handleUpdate(data);
        break;
        
      case 'position_update':
        this.handlePositionUpdate(data);
        break;
        
      case 'deletion_complete':
        this.handleDeletionComplete(data);
        break;
        
      case 'relation_update':
        this.handleRelationUpdate(data);
        break;
        
      case 'error':
        this.handleServerError(data);
        break;
        
      default:
        console.log('Unknown message type:', data.type);
    }
  }
  
  loadScene(sceneData) {
    // Clear existing data
    this.conceptCache.clear();
    this.relationCache.clear();
    this.deletedConcepts.clear();
    this.deletedRelations.clear();
    
    // Add all concepts to holographic system and cache
    sceneData.concepts.forEach(concept => {
      this.conceptCache.set(concept.id, concept);
      holographicSystem.addConcept(concept);
    });
    
    // Process relations
    sceneData.relations.forEach(relation => {
      this.relationCache.set(relation.id, relation);
      this.createRelationVisualization(relation);
    });
  }
  
  handleUpdate(update) {
    const { diff } = update;
    
    switch (diff.diff_type) {
      case 'add':
        if (update.concept) {
          this.handleConceptAddition(update.concept);
        }
        break;
        
      case 'remove':
        this.handleConceptDeletion(diff.concepts);
        break;
        
      case 'relate':
        if (update.relation) {
          this.handleRelationAddition(update.relation);
        }
        break;
        
      case 'unrelate':
        this.handleRelationDeletion(diff.relations);
        break;
        
      case 'update':
        this.handleConceptUpdate(update.concept);
        break;
    }
  }
  
  handleConceptAddition(concept) {
    this.conceptCache.set(concept.id, concept);
    holographicSystem.addConcept(concept);
    
    this.addToHistory({
      type: 'concept_added',
      timestamp: Date.now(),
      conceptId: concept.id,
      concept: JSON.parse(JSON.stringify(concept))
    });
  }
  
  handleConceptDeletion(conceptIds) {
    conceptIds.forEach(conceptId => {
      const concept = this.conceptCache.get(conceptId);
      if (concept) {
        // Store for potential undo
        this.deletedConcepts.add({
          id: conceptId,
          concept: concept,
          deletedAt: Date.now()
        });
        
        // Remove from cache
        this.conceptCache.delete(conceptId);
        
        // Remove from visualization with animation
        holographicSystem.removeConcept(conceptId, {
          animate: true,
          duration: 500,
          onComplete: () => {
            console.log(`Concept ${conceptId} removed`);
          }
        });
        
        // Remove all relations connected to this concept
        this.relationCache.forEach((relation, relationId) => {
          if (relation.source_id === conceptId || relation.target_id === conceptId) {
            this.handleRelationDeletion([relationId]);
          }
        });
        
        this.addToHistory({
          type: 'concept_deleted',
          timestamp: Date.now(),
          conceptId: conceptId,
          concept: concept
        });
      }
    });
  }
  
  handleRelationAddition(relation) {
    this.relationCache.set(relation.id, relation);
    this.createRelationVisualization(relation);
    
    this.addToHistory({
      type: 'relation_added',
      timestamp: Date.now(),
      relationId: relation.id,
      relation: JSON.parse(JSON.stringify(relation))
    });
  }
  
  handleRelationDeletion(relationIds) {
    relationIds.forEach(relationId => {
      const relation = this.relationCache.get(relationId);
      if (relation) {
        // Store for potential undo
        this.deletedRelations.add({
          id: relationId,
          relation: relation,
          deletedAt: Date.now()
        });
        
        // Remove from cache
        this.relationCache.delete(relationId);
        
        // Remove visualization
        holographicSystem.removeRelation(relationId, {
          animate: true,
          duration: 300
        });
        
        this.addToHistory({
          type: 'relation_deleted',
          timestamp: Date.now(),
          relationId: relationId,
          relation: relation
        });
      }
    });
  }
  
  handleRelationUpdate(data) {
    const { relationId, updates } = data;
    const relation = this.relationCache.get(relationId);
    
    if (relation) {
      // Update relation properties
      Object.assign(relation, updates);
      
      // Update visualization
      if (updates.strength !== undefined) {
        holographicSystem.updateRelationStrength(relationId, updates.strength);
      }
      
      if (updates.type !== undefined) {
        holographicSystem.updateRelationType(relationId, updates.type);
      }
      
      if (updates.metadata) {
        holographicSystem.updateRelationMetadata(relationId, updates.metadata);
      }
      
      this.addToHistory({
        type: 'relation_updated',
        timestamp: Date.now(),
        relationId: relationId,
        updates: updates
      });
    }
  }
  
  handleConceptUpdate(concept) {
    const existing = this.conceptCache.get(concept.id);
    if (existing) {
      // Store previous state for history
      const previousState = JSON.parse(JSON.stringify(existing));
      
      // Update cache
      Object.assign(existing, concept);
      
      // Update visualization
      holographicSystem.updateConcept(concept.id, concept);
      
      this.addToHistory({
        type: 'concept_updated',
        timestamp: Date.now(),
        conceptId: concept.id,
        previousState: previousState,
        newState: concept
      });
    }
  }
  
  handlePositionUpdate(data) {
    const concept = this.conceptCache.get(data.concept_id);
    if (concept) {
      concept.position = data.position;
      holographicSystem.updateConceptPosition(data.concept_id, data.position);
    }
  }
  
  handleDeletionComplete(data) {
    console.log(`Deletion confirmed for ${data.deletedCount} items`);
    
    // Clean up deletion tracking after confirmation
    if (data.conceptIds) {
      data.conceptIds.forEach(id => {
        this.deletedConcepts.forEach(deleted => {
          if (deleted.id === id) {
            this.deletedConcepts.delete(deleted);
          }
        });
      });
    }
  }
  
  handleServerError(data) {
    console.error('Server error:', data.error);
    
    if (data.error.includes('concept not found') && data.conceptId) {
      // Remove from local cache if server says it doesn't exist
      this.conceptCache.delete(data.conceptId);
      holographicSystem.removeConcept(data.conceptId);
    }
  }
  
  createRelationVisualization(relation) {
    const source = this.conceptCache.get(relation.source_id);
    const target = this.conceptCache.get(relation.target_id);
    
    if (source && target) {
      holographicSystem.addRelation({
        id: relation.id,
        source: source,
        target: target,
        type: relation.type,
        strength: relation.strength || 1.0,
        metadata: relation.metadata || {}
      });
    }
  }
  
  // Public API Methods
  
  addConcept(conceptData) {
    const message = {
      type: 'add_concept',
      concept: conceptData
    };
    
    if (this.offlineMode) {
      // In offline mode, add directly to cache and visualization
      const tempId = `temp_${Date.now()}`;
      conceptData.id = conceptData.id || tempId;
      this.handleConceptAddition(conceptData);
      this.queueMessage(message);
    } else {
      this.send(message);
    }
  }
  
  deleteConcept(conceptId) {
    const message = {
      type: 'delete_concept',
      concept_id: conceptId
    };
    
    if (this.offlineMode) {
      // Perform deletion locally
      this.handleConceptDeletion([conceptId]);
      this.queueMessage(message);
    } else {
      this.send(message);
    }
  }
  
  updateConcept(conceptId, updates) {
    const message = {
      type: 'update_concept',
      concept_id: conceptId,
      updates: updates
    };
    
    if (this.offlineMode) {
      // Update locally
      const concept = this.conceptCache.get(conceptId);
      if (concept) {
        Object.assign(concept, updates);
        this.handleConceptUpdate(concept);
      }
      this.queueMessage(message);
    } else {
      this.send(message);
    }
  }
  
  addRelation(sourceId, targetId, type = 'related', metadata = {}) {
    const message = {
      type: 'add_relation',
      source_id: sourceId,
      target_id: targetId,
      relation_type: type,
      metadata: metadata
    };
    
    if (this.offlineMode) {
      // Create relation locally
      const tempRelation = {
        id: `temp_rel_${Date.now()}`,
        source_id: sourceId,
        target_id: targetId,
        type: type,
        metadata: metadata
      };
      this.handleRelationAddition(tempRelation);
      this.queueMessage(message);
    } else {
      this.send(message);
    }
  }
  
  deleteRelation(relationId) {
    const message = {
      type: 'delete_relation',
      relation_id: relationId
    };
    
    if (this.offlineMode) {
      this.handleRelationDeletion([relationId]);
      this.queueMessage(message);
    } else {
      this.send(message);
    }
  }
  
  updateRelation(relationId, updates) {
    const message = {
      type: 'update_relation',
      relation_id: relationId,
      updates: updates
    };
    
    if (this.offlineMode) {
      this.handleRelationUpdate({
        relationId: relationId,
        updates: updates
      });
      this.queueMessage(message);
    } else {
      this.send(message);
    }
  }
  
  // History and Undo/Redo
  
  undo() {
    if (this.conceptHistory.length === 0) return;
    
    const lastAction = this.conceptHistory.pop();
    
    switch (lastAction.type) {
      case 'concept_added':
        this.deleteConcept(lastAction.conceptId);
        break;
        
      case 'concept_deleted':
        this.addConcept(lastAction.concept);
        break;
        
      case 'concept_updated':
        this.updateConcept(lastAction.conceptId, lastAction.previousState);
        break;
        
      case 'relation_added':
        this.deleteRelation(lastAction.relationId);
        break;
        
      case 'relation_deleted':
        const rel = lastAction.relation;
        this.addRelation(rel.source_id, rel.target_id, rel.type, rel.metadata);
        break;
    }
  }
  
  // Utility Methods
  
  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      this.queueMessage(data);
    }
  }
  
  queueMessage(message) {
    this.messageQueue.push({
      message: message,
      timestamp: Date.now()
    });
    
    // Limit queue size
    if (this.messageQueue.length > 100) {
      this.messageQueue.shift();
    }
  }
  
  flushMessageQueue() {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const { message } = this.messageQueue.shift();
      this.send(message);
    }
  }
  
  addToHistory(action) {
    this.conceptHistory.push(action);
    
    // Limit history size
    if (this.conceptHistory.length > this.maxHistorySize) {
      this.conceptHistory.shift();
    }
  }
  
  scheduleReconnect() {
    if (this.reconnectTimeout) return;
    
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
      this.connect();
    }, this.reconnectDelay);
  }
  
  // Offline Mode Support
  
  cacheScene(sceneData) {
    try {
      localStorage.setItem('conceptMeshScene', JSON.stringify({
        timestamp: Date.now(),
        scene: sceneData
      }));
    } catch (e) {
      console.warn('Failed to cache scene:', e);
    }
  }
  
  loadCachedScene() {
    try {
      const cached = localStorage.getItem('conceptMeshScene');
      if (cached) {
        const { timestamp, scene } = JSON.parse(cached);
        const age = Date.now() - timestamp;
        
        // Use cache if less than 1 hour old
        if (age < 3600000) {
          console.log('Loading cached scene');
          this.loadScene(scene);
        }
      }
    } catch (e) {
      console.warn('Failed to load cached scene:', e);
    }
  }
  
  // Search and Filter
  
  findConceptsByText(text, threshold = 0.5) {
    const results = [];
    
    this.conceptCache.forEach(concept => {
      const score = this.calculateTextSimilarity(text, concept.name + ' ' + concept.description);
      if (score > threshold) {
        results.push({ concept, score });
      }
    });
    
    return results.sort((a, b) => b.score - a.score);
  }
  
  calculateTextSimilarity(text1, text2) {
    // Simple similarity based on common words
    const words1 = text1.toLowerCase().split(/\s+/);
    const words2 = text2.toLowerCase().split(/\s+/);
    const common = words1.filter(w => words2.includes(w)).length;
    return common / Math.max(words1.length, words2.length);
  }
  
  getConnectedConcepts(conceptId, depth = 1) {
    const visited = new Set();
    const connected = [];
    
    const traverse = (id, currentDepth) => {
      if (currentDepth > depth || visited.has(id)) return;
      visited.add(id);
      
      this.relationCache.forEach(relation => {
        if (relation.source_id === id) {
          connected.push(this.conceptCache.get(relation.target_id));
          traverse(relation.target_id, currentDepth + 1);
        } else if (relation.target_id === id) {
          connected.push(this.conceptCache.get(relation.source_id));
          traverse(relation.source_id, currentDepth + 1);
        }
      });
    };
    
    traverse(conceptId, 0);
    return connected.filter(c => c !== undefined);
  }
  
  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
  
  getStatus() {
    return {
      connected: this.isConnected,
      offlineMode: this.offlineMode,
      conceptCount: this.conceptCache.size,
      relationCount: this.relationCache.size,
      queuedMessages: this.messageQueue.length,
      historySize: this.conceptHistory.length
    };
  }
}

// Export singleton
export const conceptMesh = new EnhancedConceptMeshIntegration();
