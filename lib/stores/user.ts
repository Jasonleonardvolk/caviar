// Enhanced user store with better greeting context and privacy controls
import { writable } from 'svelte/store';

export interface User {
  id: string;
  email: string;
  name: string;
  createdAt: Date;
  preferences: {
    theme: 'light' | 'dark';
    ghostPersona: string;
    memoryRetention: number;
    personalizedGreetings: boolean;
    showRecentActivity: boolean;
    privacyLevel: 'minimal' | 'balanced' | 'full';
  };
  stats: {
    conversationsCount: number;
    documentsUploaded: number;
    conceptsCreated: number;
    lastActive: Date;
    totalSessions: number;
    averageSessionLength: number;
  };
  memory: {
    lastContext?: string;
    favoriteTopics: string[];
    recentMood?: string;
    workingProjects: string[];
  };
}

export interface UserSession {
  user: User | null;
  isAuthenticated: boolean;
  sessionId: string | null;
  sessionStartTime: Date | null;
}

// Initialize user session
const initialSession: UserSession = {
  user: null,
  isAuthenticated: false,
  sessionId: null,
  sessionStartTime: null
};

export const userSession = writable<UserSession>(initialSession);

/**
 * Create a new user account with privacy-conscious defaults
 */
export function createUser(email: string, name: string): User {
  const user: User = {
    id: `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    email,
    name,
    createdAt: new Date(),
    preferences: {
      theme: 'light',
      ghostPersona: 'Mentor',
      memoryRetention: 1.0,
      personalizedGreetings: true,  // Default to on, but user can disable
      showRecentActivity: true,
      privacyLevel: 'balanced'      // Balanced privacy by default
    },
    stats: {
      conversationsCount: 0,
      documentsUploaded: 0,
      conceptsCreated: 0,
      lastActive: new Date(),
      totalSessions: 0,
      averageSessionLength: 0
    },
    memory: {
      favoriteTopics: [],
      workingProjects: []
    }
  };

  saveUserToMemory(user);
  return user;
}

/**
 * Login user with session tracking
 */
export function loginUser(email: string): boolean {
  try {
    const savedUsers = getSavedUsers();
    const user = savedUsers.find(u => u.email === email);
    
    if (user) {
      const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const sessionStartTime = new Date();
      
      // Update session stats
      user.stats.lastActive = sessionStartTime;
      user.stats.totalSessions += 1;
      saveUserToMemory(user);
      
      userSession.set({
        user,
        isAuthenticated: true,
        sessionId,
        sessionStartTime
      });
      
      // Save session for auto-login
      saveCurrentSession(email);
      
      console.log('✅ User logged in successfully:', email);
      return true;
    }
    
    return false;
  } catch (error) {
    console.error('Failed to login user:', error);
    return false;
  }
}

/**
 * Register new user
 */
export function registerUser(email: string, name: string): boolean {
  try {
    const savedUsers = getSavedUsers();
    
    // Check if user already exists
    if (savedUsers.some(u => u.email === email)) {
      console.warn('User already exists:', email);
      return false;
    }
    
    const user = createUser(email, name);
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const sessionStartTime = new Date();
    
    userSession.set({
      user,
      isAuthenticated: true,
      sessionId,
      sessionStartTime
    });
    
    // Save session for auto-login
    saveCurrentSession(email);
    
    console.log('✅ User registered successfully:', email);
    return true;
  } catch (error) {
    console.error('Failed to register user:', error);
    return false;
  }
}

/**
 * Logout user with session length tracking
 */
export function logoutUser() {
  userSession.update(session => {
    if (session.user && session.sessionStartTime) {
      // Calculate session length
      const sessionLength = (Date.now() - session.sessionStartTime.getTime()) / (1000 * 60); // minutes
      
      // Update average session length
      const totalSessions = session.user.stats.totalSessions;
      const currentAverage = session.user.stats.averageSessionLength;
      const newAverage = ((currentAverage * (totalSessions - 1)) + sessionLength) / totalSessions;
      
      session.user.stats.averageSessionLength = newAverage;
      saveUserToMemory(session.user);
    }
    
    return {
      user: null,
      isAuthenticated: false,
      sessionId: null,
      sessionStartTime: null
    };
  });
  
  // Clear session
  localStorage.removeItem('tori-current-session');
  
  console.log('👋 User logged out');
}

/**
 * Update user stats with proper increment handling
 */
export function updateUserStats(updates: Partial<User['stats']>) {
  userSession.update(session => {
    if (session.user) {
      const currentStats = session.user.stats;
      const updatedStats = { ...currentStats };
      
      // Handle increments
      if ('conversationsCount' in updates) {
        updatedStats.conversationsCount = currentStats.conversationsCount + 1;
      }
      if ('documentsUploaded' in updates) {
        updatedStats.documentsUploaded = currentStats.documentsUploaded + 1;
      }
      if ('conceptsCreated' in updates) {
        updatedStats.conceptsCreated = currentStats.conceptsCreated + 1;
      }
      
      // Apply other updates
      Object.assign(updatedStats, updates);
      updatedStats.lastActive = new Date();
      
      const updatedUser = {
        ...session.user,
        stats: updatedStats
      };
      
      saveUserToMemory(updatedUser);
      
      return {
        ...session,
        user: updatedUser
      };
    }
    return session;
  });
}

/**
 * Update user preferences
 */
export function updateUserPreferences(updates: Partial<User['preferences']>) {
  userSession.update(session => {
    if (session.user) {
      const updatedUser = {
        ...session.user,
        preferences: {
          ...session.user.preferences,
          ...updates
        }
      };
      
      saveUserToMemory(updatedUser);
      
      return {
        ...session,
        user: updatedUser
      };
    }
    return session;
  });
}

/**
 * Update user memory context (for personalized greetings)
 */
export function updateUserMemory(updates: Partial<User['memory']>) {
  userSession.update(session => {
    if (session.user) {
      const updatedUser = {
        ...session.user,
        memory: {
          ...session.user.memory,
          ...updates
        }
      };
      
      saveUserToMemory(updatedUser);
      
      return {
        ...session,
        user: updatedUser
      };
    }
    return session;
  });
}

/**
 * Add a favorite topic (for personalization)
 */
export function addFavoriteTopic(topic: string) {
  userSession.update(session => {
    if (session.user) {
      const currentTopics = session.user.memory.favoriteTopics || [];
      const updatedTopics = [...new Set([...currentTopics, topic])].slice(0, 10); // Keep top 10
      
      updateUserMemory({ favoriteTopics: updatedTopics });
    }
    return session;
  });
}

/**
 * Update working project context
 */
export function updateWorkingProject(project: string) {
  userSession.update(session => {
    if (session.user) {
      const currentProjects = session.user.memory.workingProjects || [];
      const updatedProjects = [project, ...currentProjects.filter(p => p !== project)].slice(0, 5); // Keep top 5
      
      updateUserMemory({ workingProjects: updatedProjects });
    }
    return session;
  });
}

/**
 * Save user to memory system
 */
function saveUserToMemory(user: User) {
  try {
    const savedUsers = getSavedUsers();
    const existingIndex = savedUsers.findIndex(u => u.id === user.id);
    
    if (existingIndex >= 0) {
      savedUsers[existingIndex] = user;
    } else {
      savedUsers.push(user);
    }
    
    localStorage.setItem('tori-users', JSON.stringify(savedUsers));
    console.log('💾 User saved to memory system');
  } catch (error) {
    console.warn('Failed to save user:', error);
  }
}

/**
 * Get all saved users
 */
function getSavedUsers(): User[] {
  try {
    const saved = localStorage.getItem('tori-users');
    if (saved) {
      return JSON.parse(saved).map((user: any) => ({
        ...user,
        createdAt: new Date(user.createdAt),
        stats: {
          ...user.stats,
          lastActive: new Date(user.stats.lastActive)
        }
      }));
    }
    return [];
  } catch (error) {
    console.warn('Failed to load users:', error);
    return [];
  }
}

/**
 * Auto-login if session exists
 */
export function initializeUserSession() {
  try {
    const savedSession = localStorage.getItem('tori-current-session');
    if (savedSession) {
      const session = JSON.parse(savedSession);
      
      // Validate session is still valid (less than 7 days old for privacy)
      const sessionTime = new Date(session.timestamp);
      const now = new Date();
      const daysDiff = (now.getTime() - sessionTime.getTime()) / (1000 * 60 * 60 * 24);
      
      if (daysDiff < 7 && session.email) {
        loginUser(session.email);
      } else {
        // Clear expired session
        localStorage.removeItem('tori-current-session');
      }
    }
  } catch (error) {
    console.warn('Failed to initialize user session:', error);
  }
}

/**
 * Save current session for auto-login (privacy-conscious duration)
 */
export function saveCurrentSession(email: string) {
  try {
    const sessionData = {
      email,
      timestamp: new Date().toISOString()
    };
    localStorage.setItem('tori-current-session', JSON.stringify(sessionData));
  } catch (error) {
    console.warn('Failed to save session:', error);
  }
}