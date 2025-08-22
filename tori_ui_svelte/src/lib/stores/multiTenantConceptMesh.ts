import { writable, derived, get } from 'svelte/store';
import type { ConceptDiff, SystemState, ConceptNode } from './types';

// ===================================================================
// TORI MULTI-TENANT CONCEPT MESH - Three-Tier Architecture
// Production Ready: June 4, 2025
// Features: Private → Organization → Foundation Knowledge
// Authentication: JWT-based user sessions
// Architecture: File-based storage with tier inheritance
// ===================================================================

// 🏢 NEW: Multi-tenant types
export interface UserProfile {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'member' | 'viewer';
  organization_ids: string[];
  created_at: string;
  last_active: string;
  concept_count: number;
  preferences: Record<string, any>;
}

export interface Organization {
  id: string;
  name: string;
  description: string;
  created_at: string;
  concept_count: number;
}

export interface AuthSession {
  token: string;
  user: UserProfile;
  expires_at: string;
  authenticated: boolean;
}

export type KnowledgeTier = 'private' | 'organization' | 'foundation';

export interface TieredConcept {
  name: string;
  confidence: number;
  context: string;
  tier: KnowledgeTier;
  owner_id: string;
  source_document?: string;
  tags: string[];
  created_at: string;
  access_count: number;
  metadata: Record<string, any>;
}

export interface KnowledgeStats {
  tiers: {
    private: { concepts: number; domains: string[]; recent_activity: number };
    organization: { concepts: number; domains: string[]; recent_activity: number };
    foundation: { concepts: number; domains: string[]; recent_activity: number };
  };
  total_concepts: number;
  total_domains: string[];
  user_activity: {
    concepts_created: number;
    concepts_accessed: number;
    last_activity: string | null;
  };
}

// Enhanced concept structure extending existing ProductionConcept
export interface MultiTenantConcept extends TieredConcept {
  eigenfunction_id: string;
  cluster_id?: number | string;
  title?: string;
  timestamp?: string;
  strength?: number;
  type?: string;
  
  // Cognitive system fields (inherited from existing system)
  solitonPhase?: number;
  holographicPosition?: { x: number; y: number; z: number };
  braidLoopId?: string;
  ghostPersona?: string;
  
  // Pipeline metadata
  purity?: number;
  frequency?: number;
  section?: string;
  paragraph?: string;
  summary?: string;
  domain?: string;
  
  // Quality metrics
  qualityScore?: number;
  expertiseLevel?: 'novice' | 'intermediate' | 'expert' | 'specialist';
  contradictionLevel?: number;
  coherenceContribution?: number;
}

// Enhanced ConceptDiff with multi-tenant awareness
export interface MultiTenantConceptDiff extends ConceptDiff {
  tier: KnowledgeTier;
  owner_id: string;
  tieredConcepts?: MultiTenantConcept[];
  
  // Enhanced metadata
  userContext?: {
    user_id: string;
    username: string;
    organization_id?: string;
  };
  
  // Tier-specific metadata
  tierMetadata?: {
    storage_location: string;
    access_permissions: string[];
    inheritance_chain: KnowledgeTier[];
  };
}

// ===================================================================
// AUTHENTICATION STATE MANAGEMENT
// ===================================================================

const createAuthStore = () => {
  const { subscribe, set, update } = writable<AuthSession | null>(null);
  
  return {
    subscribe,
    
    // Load session from localStorage
    loadSession: () => {
      try {
        const stored = localStorage.getItem('tori-auth-session');
        if (stored) {
          const session = JSON.parse(stored);
          
          // Check if session is expired
          const expiresAt = new Date(session.expires_at);
          if (expiresAt > new Date()) {
            set(session);
            console.log('🔐 Session restored:', session.user.username);
            return session;
          } else {
            // Session expired, clear it
            localStorage.removeItem('tori-auth-session');
            console.log('🔐 Session expired, cleared');
          }
        }
        return null;
      } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
        console.warn('Failed to load auth session:', error);
        localStorage.removeItem('tori-auth-session');
        return null;
      
}
    },
    
    // Save session
    setSession: (session: AuthSession) => {
      try {
        localStorage.setItem('tori-auth-session', JSON.stringify(session));
        set(session);
        console.log('🔐 Session saved:', session.user.username);
      } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
        console.error('Failed to save auth session:', error);
      
}
    },
    
    // Clear session
    clearSession: () => {
      try {
        localStorage.removeItem('tori-auth-session');
        set(null);
        console.log('🔐 Session cleared');
      } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
        console.error('Failed to clear auth session:', error);
      
}
    },
    
    // Update user info
    updateUser: (user: UserProfile) => {
      update(session => {
        if (session) {
          const updated = { ...session, user };
          localStorage.setItem('tori-auth-session', JSON.stringify(updated));
          return updated;
        }
        return session;
      });
    }
  };
};

export const authSession = createAuthStore();

// Derived stores for easy access
export const currentUser = derived(authSession, $auth => $auth?.user || null);
export const isAuthenticated = derived(authSession, $auth => $auth?.authenticated || false);
export const userOrganizations = writable<Organization[]>([]);

// ===================================================================
// MULTI-TENANT CONCEPT STORAGE
// ===================================================================

const MULTI_TENANT_STORAGE_KEY = 'tori-multi-tenant-concepts';
const KNOWLEDGE_STATS_KEY = 'tori-knowledge-stats';
const VERSION = '1.0.0-multi-tenant';

// Core stores (extending existing)
export const multiTenantConceptMesh = writable<MultiTenantConceptDiff[]>([]);
export const knowledgeStats = writable<KnowledgeStats>({
  tiers: {
    private: { concepts: 0, domains: [], recent_activity: 0 },
    organization: { concepts: 0, domains: [], recent_activity: 0 },
    foundation: { concepts: 0, domains: [], recent_activity: 0 }
  },
  total_concepts: 0,
  total_domains: [],
  user_activity: {
    concepts_created: 0,
    concepts_accessed: 0,
    last_activity: null
  }
});

// API configuration
const API_BASE_URL = 'http://localhost:8002'; // Will be dynamic from port config

// ===================================================================
// API COMMUNICATION LAYER
// ===================================================================

class MultiTenantAPI {
  private baseUrl: string;
  
  constructor() {
    this.baseUrl = API_BASE_URL;
    this.loadApiConfig();
  }
  
  private async loadApiConfig() {
    try {
      const response = await fetch('/api_port.json');
      const config = await response.json();
      this.baseUrl = config.api_url || API_BASE_URL;
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.warn('Could not load API config, using default:', this.baseUrl);
    
}
  }
  
  private getAuthHeaders(): Record<string, string> {
    const session = get(authSession);
    if (session?.token) {
      return {
        'Authorization': `Bearer ${session.token}`,
        'Content-Type': 'application/json'
      };
    }
    return { 'Content-Type': 'application/json' };
  }
  
  // Authentication
  async login(username: string, password: string): Promise<AuthSession | null> {
    try {
      const response = await fetch(`${this.baseUrl}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      
      const data = await response.json();
      
      if (data.success) {
        const session: AuthSession = {
          token: data.token,
          user: data.user,
          expires_at: data.expires_at,
          authenticated: true
        };
        
        authSession.setSession(session);
        return session;
      } else {
        throw new Error(data.message || 'Login failed');
      }
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.error('Login error:', error);
      throw error;
    
}
  }
  
  async register(username: string, email: string, password: string): Promise<UserProfile> {
    try {
      const response = await fetch(`${this.baseUrl}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password })
      });
      
      const data = await response.json();
      
      if (data.success) {
        return data.user;
      } else {
        throw new Error(data.message || 'Registration failed');
      }
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.error('Registration error:', error);
      throw error;
    
}
  }
  
  async logout(): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/auth/logout`, {
        method: 'POST',
        headers: this.getAuthHeaders()
      });
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.warn('Logout API call failed:', error);
    
} finally {
      authSession.clearSession();
    }
  }
  
  // Knowledge management
  async searchConcepts(query: string, maxResults: number = 20): Promise<TieredConcept[]> {
    try {
      const response = await fetch(`${this.baseUrl}/knowledge/search`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({
          query,
          max_results: maxResults,
          include_tiers: ['private', 'organization', 'foundation']
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        return data.concepts || [];
      } else {
        throw new Error('Search failed');
      }
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.error('Concept search error:', error);
      return [];
    
}
  }
  
  async getKnowledgeStats(): Promise<KnowledgeStats | null> {
    try {
      const response = await fetch(`${this.baseUrl}/knowledge/stats`, {
        headers: this.getAuthHeaders()
      });
      
      const data = await response.json();
      
      if (data.success) {
        return data.stats;
      }
      return null;
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.error('Knowledge stats error:', error);
      return null;
    
}
  }
  
  // Organizations
  async getUserOrganizations(): Promise<Organization[]> {
    try {
      const response = await fetch(`${this.baseUrl}/organizations`, {
        headers: this.getAuthHeaders()
      });
      
      const data = await response.json();
      
      if (data.success) {
        return data.organizations || [];
      }
      return [];
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.error('Get organizations error:', error);
      return [];
    
}
  }
  
  async createOrganization(name: string, description: string): Promise<Organization | null> {
    try {
      const response = await fetch(`${this.baseUrl}/organizations`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({ name, description })
      });
      
      const data = await response.json();
      
      if (data.success) {
        return data.organization;
      } else {
        throw new Error(data.message || 'Organization creation failed');
      }
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.error('Create organization error:', error);
      throw error;
    
}
  }
  
  // Enhanced PDF extraction with tier support
  async extractPDF(filePath: string, filename: string, contentType: string, 
                  tier: KnowledgeTier = 'private', organizationId?: string, 
                  progressId?: string): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/extract`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({
          file_path: filePath,
          filename,
          content_type: contentType,
          tier,
          organization_id: organizationId,
          progress_id: progressId
        })
      });
      
      return await response.json();
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.error('PDF extraction error:', error);
      throw error;
    
}
  }
  
  // Enhanced chat with multi-tenant support
  async chat(message: string, userId?: string): Promise<any> {
    try {
      const currentSession = get(authSession);
      const actualUserId = userId || currentSession?.user?.id || 'anonymous';
      
      const response = await fetch(`${this.baseUrl}/chat`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({
          message,
          userId: actualUserId,
          timestamp: new Date().toISOString()
        })
      });
      
      return await response.json();
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.error('Chat error:', error);
      throw error;
    
}
  }
}

// Global API instance
export const multiTenantAPI = new MultiTenantAPI();

// ===================================================================
// ENHANCED CONCEPT MANAGEMENT WITH MULTI-TENANT SUPPORT
// ===================================================================

/**
 * Add concept diff with multi-tenant awareness
 */
export function addMultiTenantConceptDiff(
  docId: string, 
  concepts: any, 
  tier: KnowledgeTier = 'private',
  metadata?: Record<string, any>
) {
  const user = get(currentUser);
  if (!user && tier !== 'foundation') {
    console.warn('❌ User authentication required for non-foundation concepts');
    return;
  }
  
  try {
    const now = new Date();
    const diffId = `mt_diff_${tier}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Validate and normalize concepts
    const validatedConcepts = validateMultiTenantConcepts(concepts, tier, user?.id || 'foundation');
    
    if (validatedConcepts.length === 0) {
      console.warn('❌ No valid concepts to add');
      return;
    }
    
    // Create multi-tenant concept diff
    const multiTenantDiff: MultiTenantConceptDiff = {
      id: diffId,
      type: 'document',
      title: docId,
      concepts: validatedConcepts.map(c => c.name),
      summary: `${tier.toUpperCase()}: ${validatedConcepts.length} concepts from "${docId}"`,
      timestamp: now,
      tier,
      owner_id: user?.id || 'foundation',
      tieredConcepts: validatedConcepts,
      metadata: {
        ...metadata,
        version: VERSION,
        tier,
        owner_id: user?.id || 'foundation',
        multi_tenant: true,
        concept_count: validatedConcepts.length
      },
      userContext: user ? {
        user_id: user.id,
        username: user.username,
        organization_id: metadata?.organization_id
      } : undefined,
      tierMetadata: {
        storage_location: `${tier}/${user?.id || 'foundation'}`,
        access_permissions: generateAccessPermissions(tier, user),
        inheritance_chain: getInheritanceChain(tier)
      }
    };
    
    // Update concept mesh
    multiTenantConceptMesh.update(current => {
      const updated = [...current, multiTenantDiff];
      saveMultiTenantState(updated);
      return updated;
    });
    
    // Update knowledge stats
    updateKnowledgeStats(validatedConcepts, tier);
    
    // Dispatch event for UI updates
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tori:multi-tenant-concept-diff', {
        detail: { 
          diff: multiTenantDiff, 
          concepts: validatedConcepts,
          tier,
          user: user?.username || 'anonymous'
        }
      }));
    }
    
    console.log(`✅ Multi-tenant concept diff added: ${validatedConcepts.length} concepts in ${tier} tier`);
    
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.error('❌ Failed to add multi-tenant concept diff:', error);
  
}
}

/**
 * Validate and normalize concepts for multi-tenant storage
 */
function validateMultiTenantConcepts(concepts: any, tier: KnowledgeTier, ownerId: string): MultiTenantConcept[] {
  if (!Array.isArray(concepts)) {
    return [];
  }
  
  return concepts.map((concept, index) => {
    const now = new Date().toISOString();
    
    if (typeof concept === 'string') {
      return {
        eigenfunction_id: `${tier}_${index}_${Date.now()}`,
        name: concept,
        confidence: 0.7,
        context: `${tier} tier concept`,
        tier,
        owner_id: ownerId,
        tags: [],
        created_at: now,
        access_count: 0,
        metadata: { extraction_method: 'multi_tenant_pipeline' }
      };
    }
    
    if (typeof concept === 'object' && concept !== null) {
      return {
        eigenfunction_id: concept.eigenfunction_id || `${tier}_${index}_${Date.now()}`,
        name: concept.name || `Concept ${index}`,
        confidence: concept.confidence || concept.score || 0.7,
        context: concept.context || `${tier} tier concept`,
        tier,
        owner_id: ownerId,
        tags: concept.tags || [],
        created_at: now,
        access_count: 0,
        metadata: concept.metadata || {},
        
        // Enhanced fields
        cluster_id: concept.cluster_id,
        title: concept.title,
        timestamp: concept.timestamp,
        strength: concept.strength || concept.confidence,
        type: concept.type || tier,
        
        // Quality metrics
        qualityScore: concept.qualityScore || calculateConceptQuality(concept),
        expertiseLevel: concept.expertiseLevel || determineExpertiseLevel(concept),
        contradictionLevel: concept.contradictionLevel || 0,
        coherenceContribution: concept.coherenceContribution || 0.1,
        
        // Pipeline metadata
        purity: concept.purity,
        frequency: concept.frequency,
        section: concept.section,
        paragraph: concept.paragraph,
        summary: concept.summary,
        domain: concept.domain,
        
        // Cognitive fields
        solitonPhase: concept.solitonPhase || Math.random() * 2 * Math.PI,
        holographicPosition: concept.holographicPosition || {
          x: (Math.random() - 0.5) * 4,
          y: (Math.random() - 0.5) * 4,
          z: (Math.random() - 0.5) * 4
        },
        braidLoopId: concept.braidLoopId,
        ghostPersona: concept.ghostPersona
      };
    }
    
    return {
      eigenfunction_id: `${tier}_fallback_${index}_${Date.now()}`,
      name: `Unknown Concept ${index}`,
      confidence: 0.1,
      context: 'Fallback concept',
      tier,
      owner_id: ownerId,
      tags: [],
      created_at: now,
      access_count: 0,
      metadata: { error: 'Invalid concept format' }
    };
  }).filter(concept => concept.name && concept.name.trim().length > 0);
}

/**
 * Generate access permissions based on tier and user
 */
function generateAccessPermissions(tier: KnowledgeTier, user: UserProfile | null): string[] {
  const permissions = [];
  
  if (tier === 'private' && user) {
    permissions.push(`user:${user.id}:read`, `user:${user.id}:write`);
  } else if (tier === 'organization' && user) {
    user.organization_ids.forEach(orgId => {
      permissions.push(`org:${orgId}:read`);
      if (user.role === 'admin' || user.role === 'member') {
        permissions.push(`org:${orgId}:write`);
      }
    });
  } else if (tier === 'foundation') {
    permissions.push('foundation:read');
    if (user?.role === 'admin') {
      permissions.push('foundation:write');
    }
  }
  
  return permissions;
}

/**
 * Get inheritance chain for tier
 */
function getInheritanceChain(tier: KnowledgeTier): KnowledgeTier[] {
  switch (tier) {
    case 'private':
      return ['private', 'organization', 'foundation'];
    case 'organization':
      return ['organization', 'foundation'];
    case 'foundation':
      return ['foundation'];
    default:
      return ['foundation'];
  }
}

/**
 * Update knowledge statistics
 */
function updateKnowledgeStats(concepts: MultiTenantConcept[], tier: KnowledgeTier) {
  knowledgeStats.update(current => {
    const updated = { ...current };
    
    // Update tier stats
    updated.tiers[tier].concepts += concepts.length;
    updated.tiers[tier].recent_activity += 1;
    
    // Update domains
    const newDomains = new Set(updated.tiers[tier].domains);
    concepts.forEach(concept => {
      if (concept.domain) {
        newDomains.add(concept.domain);
      }
    });
    updated.tiers[tier].domains = Array.from(newDomains);
    
    // Update totals
    updated.total_concepts = Object.values(updated.tiers).reduce((sum, tier) => sum + tier.concepts, 0);
    
    const allDomains = new Set<string>();
    Object.values(updated.tiers).forEach(tier => {
      tier.domains.forEach(domain => allDomains.add(domain));
    });
    updated.total_domains = Array.from(allDomains);
    
    // Update user activity
    updated.user_activity.concepts_created += concepts.length;
    updated.user_activity.last_activity = new Date().toISOString();
    
    // Save stats
    saveKnowledgeStats(updated);
    
    return updated;
  });
}

// ===================================================================
// MULTI-TENANT SEARCH FUNCTIONALITY
// ===================================================================

/**
 * Search concepts across all accessible tiers
 */
export async function searchMultiTenantConcepts(query: string, maxResults: number = 20): Promise<TieredConcept[]> {
  const user = get(currentUser);
  
  try {
    if (user) {
      // Authenticated user: search via API
      return await multiTenantAPI.searchConcepts(query, maxResults);
    } else {
      // Anonymous user: search local foundation concepts only
      return searchLocalFoundationConcepts(query, maxResults);
    }
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.error('Multi-tenant search error:', error);
    return [];
  
}
}

/**
 * Search local foundation concepts for anonymous users
 */
function searchLocalFoundationConcepts(query: string, maxResults: number): TieredConcept[] {
  const mesh = get(multiTenantConceptMesh);
  const queryLower = query.toLowerCase();
  const results: TieredConcept[] = [];
  
  mesh.forEach(diff => {
    if (diff.tier === 'foundation' && diff.tieredConcepts) {
      diff.tieredConcepts.forEach(concept => {
        if (concept.name.toLowerCase().includes(queryLower) ||
            concept.context.toLowerCase().includes(queryLower)) {
          results.push(concept);
        }
      });
    }
  });
  
  return results.slice(0, maxResults);
}

// ===================================================================
// STATE PERSISTENCE
// ===================================================================

/**
 * Save multi-tenant state to localStorage
 */
function saveMultiTenantState(diffs: MultiTenantConceptDiff[]): void {
  try {
    const payload = {
      version: VERSION,
      timestamp: new Date().toISOString(),
      diffs: diffs,
      user_id: get(currentUser)?.id || null
    };
    
    localStorage.setItem(MULTI_TENANT_STORAGE_KEY, JSON.stringify(payload));
    console.log(`💾 Multi-tenant state saved: ${diffs.length} diffs`);
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.warn('Failed to save multi-tenant state:', error);
  
}
}

/**
 * Load multi-tenant state from localStorage
 */
function loadMultiTenantState(): void {
  try {
    const stored = localStorage.getItem(MULTI_TENANT_STORAGE_KEY);
    if (!stored) return;
    
    const parsed = JSON.parse(stored);
    if (parsed.diffs && Array.isArray(parsed.diffs)) {
      const restoredDiffs = parsed.diffs.map((diff: any) => ({
        ...diff,
        timestamp: new Date(diff.timestamp)
      }));
      
      multiTenantConceptMesh.set(restoredDiffs);
      console.log(`📊 Multi-tenant state restored: ${restoredDiffs.length} diffs`);
    }
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.warn('Failed to load multi-tenant state:', error);
    localStorage.removeItem(MULTI_TENANT_STORAGE_KEY);
  
}
}

/**
 * Save knowledge statistics
 */
function saveKnowledgeStats(stats: KnowledgeStats): void {
  try {
    localStorage.setItem(KNOWLEDGE_STATS_KEY, JSON.stringify(stats));
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.warn('Failed to save knowledge stats:', error);
  
}
}

/**
 * Load knowledge statistics
 */
function loadKnowledgeStats(): void {
  try {
    const stored = localStorage.getItem(KNOWLEDGE_STATS_KEY);
    if (stored) {
      const stats = JSON.parse(stored);
      knowledgeStats.set(stats);
    }
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.warn('Failed to load knowledge stats:', error);
  
}
}

// ===================================================================
// INITIALIZATION AND SYNC
// ===================================================================

/**
 * Initialize multi-tenant system
 */
export async function initializeMultiTenantSystem() {
  console.log('🏢 Initializing Multi-Tenant System...');
  
  // Load persisted state
  loadMultiTenantState();
  loadKnowledgeStats();
  
  // Load auth session
  authSession.loadSession();
  
  // If authenticated, sync with server
  const user = get(currentUser);
  if (user) {
    await syncWithServer();
  }
  
  console.log('✅ Multi-tenant system initialized');
}

/**
 * Sync local state with server
 */
export async function syncWithServer() {
  const user = get(currentUser);
  if (!user) return;
  
  try {
    console.log('🔄 Syncing with server...');
    
    // Get knowledge stats from server
    const stats = await multiTenantAPI.getKnowledgeStats();
    if (stats) {
      knowledgeStats.set(stats);
      saveKnowledgeStats(stats);
    }
    
    // Get user organizations
    const organizations = await multiTenantAPI.getUserOrganizations();
    userOrganizations.set(organizations);
    
    console.log('✅ Server sync complete');
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.error('❌ Server sync failed:', error);
  
}
}

/**
 * Clear all multi-tenant data
 */
export function clearMultiTenantData() {
  multiTenantConceptMesh.set([]);
  knowledgeStats.set({
    tiers: {
      private: { concepts: 0, domains: [], recent_activity: 0 },
      organization: { concepts: 0, domains: [], recent_activity: 0 },
      foundation: { concepts: 0, domains: [], recent_activity: 0 }
    },
    total_concepts: 0,
    total_domains: [],
    user_activity: {
      concepts_created: 0,
      concepts_accessed: 0,
      last_activity: null
    }
  });
  
  localStorage.removeItem(MULTI_TENANT_STORAGE_KEY);
  localStorage.removeItem(KNOWLEDGE_STATS_KEY);
  
  console.log('🗑️ Multi-tenant data cleared');
}

// ===================================================================
// UTILITY FUNCTIONS
// ===================================================================

function calculateConceptQuality(concept: any): number {
  let quality = 0.5;
  
  if (concept.purity && concept.purity > 0.7) quality += 0.2;
  if (concept.context && concept.context.length > 50) quality += 0.1;
  if (concept.section || concept.paragraph) quality += 0.1;
  if (concept.frequency && concept.frequency > 2) quality += 0.1;
  if (concept.domain && concept.domain !== 'general') quality += 0.1;
  
  return Math.min(1.0, quality);
}

function determineExpertiseLevel(concept: any): 'novice' | 'intermediate' | 'expert' | 'specialist' {
  const score = concept.score || concept.purity || 0.5;
  const hasContext = concept.context && concept.context.length > 100;
  const hasDomain = concept.domain && concept.domain !== 'general';
  
  if (score > 0.9 && hasContext && hasDomain) return 'specialist';
  if (score > 0.8 && hasContext) return 'expert';
  if (score > 0.6) return 'intermediate';
  return 'novice';
}

// ===================================================================
// DERIVED STORES AND COMPUTED VALUES
// ===================================================================

export const privateConcepts = derived(multiTenantConceptMesh, $mesh => 
  $mesh.filter(diff => diff.tier === 'private')
);

export const organizationConcepts = derived(multiTenantConceptMesh, $mesh => 
  $mesh.filter(diff => diff.tier === 'organization')
);

export const foundationConcepts = derived(multiTenantConceptMesh, $mesh => 
  $mesh.filter(diff => diff.tier === 'foundation')
);

export const totalConceptCount = derived(knowledgeStats, $stats => $stats.total_concepts);

export const userCanCreateOrganization = derived(currentUser, $user => 
  $user?.role === 'admin' || $user?.role === 'member'
);

export const userCanAccessFoundation = derived(currentUser, $user => 
  $user?.role === 'admin'
);

// ===================================================================
// INITIALIZATION
// ===================================================================

// Auto-initialize when module loads
if (typeof window !== 'undefined') {
  initializeMultiTenantSystem();
  
  // Add global debug helpers
  (window as any).multiTenantAPI = multiTenantAPI;
  (window as any).clearMultiTenantData = clearMultiTenantData;
  (window as any).syncWithServer = syncWithServer;
  
  console.log('🏢 TORI Multi-Tenant ConceptMesh initialized');
  console.log('🎯 Features: Three-tier knowledge, JWT auth, organization support');
}

// Export compatibility with existing conceptMesh
export const conceptMesh = multiTenantConceptMesh;
export const addConceptDiff = addMultiTenantConceptDiff;

console.log('✅ TORI Multi-Tenant ConceptMesh - FULLY LOADED');
console.log('🏢 Architecture: Private → Organization → Foundation');
console.log('🔐 Authentication: JWT-based sessions');
console.log('🚀 Ready for multi-tenant operations');
