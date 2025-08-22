/**
 * Typing Prosody Analyzer
 * =======================
 * 
 * Analyzes user typing patterns to understand:
 * - Emotional state
 * - Confidence levels
 * - Cognitive load
 * - Intent clarity
 */

import { writable, derived } from 'svelte/store';

export interface TypingEvent {
  key: string;
  timestamp: number;
  position: number;
  action: 'keydown' | 'keyup' | 'delete' | 'paste';
}

export interface ProsodyMetrics {
  // Basic metrics
  wpm: number;                    // Words per minute
  charactersPerMinute: number;    // Raw typing speed
  
  // Pause analysis
  averagePauseDuration: number;   // Milliseconds
  pauseFrequency: number;         // Pauses per minute
  longPauses: number;            // Pauses > 2 seconds
  
  // Deletion patterns
  deleteRatio: number;           // Deletes / total keystrokes
  backspaceStreak: number;       // Max consecutive backspaces
  
  // Rhythm analysis
  burstiness: number;            // 0-1 (steady vs bursts)
  rhythmVariance: number;        // Typing rhythm consistency
  
  // Advanced metrics
  hesitationScore: number;       // 0-1 (confident vs uncertain)
  emotionalIntensity: number;    // 0-1 (calm vs intense)
  cognitiveLoad: number;         // 0-1 (easy vs struggling)
}

export interface ProsodyInsight {
  type: 'emotion' | 'confidence' | 'intent' | 'cognitive';
  label: string;
  confidence: number;
  description: string;
}

class TypingProsodyAnalyzer {
  private events: TypingEvent[] = [];
  private sessionStart: number = Date.now();
  private lastEventTime: number = Date.now();
  private deleteStreak: number = 0;
  private maxDeleteStreak: number = 0;
  private pauseThreshold: number = 1000; // 1 second
  private longPauseThreshold: number = 2000; // 2 seconds
  
  // Stores
  public metrics = writable<ProsodyMetrics>(this.getEmptyMetrics());
  public insights = writable<ProsodyInsight[]>([]);
  
  // Derived stores
  public isHesitating = derived(this.metrics, $m => $m.hesitationScore > 0.6);
  public isIntense = derived(this.metrics, $m => $m.emotionalIntensity > 0.7);
  public isStruggling = derived(this.metrics, $m => $m.cognitiveLoad > 0.7);

  private getEmptyMetrics(): ProsodyMetrics {
    return {
      wpm: 0,
      charactersPerMinute: 0,
      averagePauseDuration: 0,
      pauseFrequency: 0,
      longPauses: 0,
      deleteRatio: 0,
      backspaceStreak: 0,
      burstiness: 0,
      rhythmVariance: 0,
      hesitationScore: 0,
      emotionalIntensity: 0,
      cognitiveLoad: 0
    };
  }

  recordKeystroke(event: KeyboardEvent, currentText: string): void {
    const now = Date.now();
    const timeSinceLastEvent = now - this.lastEventTime;
    
    // Detect pause
    if (timeSinceLastEvent > this.pauseThreshold) {
      this.recordPause(timeSinceLastEvent);
    }
    
    // Record event
    const typingEvent: TypingEvent = {
      key: event.key,
      timestamp: now,
      position: currentText.length,
      action: this.getAction(event)
    };
    
    this.events.push(typingEvent);
    this.lastEventTime = now;
    
    // Update delete streak
    if (event.key === 'Backspace' || event.key === 'Delete') {
      this.deleteStreak++;
      this.maxDeleteStreak = Math.max(this.maxDeleteStreak, this.deleteStreak);
    } else {
      this.deleteStreak = 0;
    }
    
    // Clean old events (keep last 5 minutes)
    const fiveMinutesAgo = now - 300000;
    this.events = this.events.filter(e => e.timestamp > fiveMinutesAgo);
    
    // Update metrics
    this.updateMetrics();
  }

  recordPause(duration: number): void {
    // Pauses indicate thinking, uncertainty, or careful consideration
    const pauseEvent: TypingEvent = {
      key: 'PAUSE',
      timestamp: Date.now(),
      position: -1,
      action: 'keydown'
    };
    
    this.events.push(pauseEvent);
  }

  private getAction(event: KeyboardEvent): TypingEvent['action'] {
    if (event.key === 'Backspace' || event.key === 'Delete') {
      return 'delete';
    }
    if (event.ctrlKey && event.key === 'v') {
      return 'paste';
    }
    return event.type as 'keydown' | 'keyup';
  }

  private updateMetrics(): void {
    const now = Date.now();
    const sessionDuration = (now - this.sessionStart) / 1000; // seconds
    
    if (this.events.length < 10 || sessionDuration < 5) {
      // Not enough data yet
      return;
    }
    
    // Calculate basic metrics
    const charEvents = this.events.filter(e => e.action === 'keydown' && e.key.length === 1);
    const deleteEvents = this.events.filter(e => e.action === 'delete');
    const pauseEvents = this.events.filter(e => e.key === 'PAUSE');
    
    // WPM calculation (assume 5 chars per word)
    const totalChars = charEvents.length;
    const wpm = (totalChars / 5) / (sessionDuration / 60);
    const cpm = totalChars / (sessionDuration / 60);
    
    // Pause analysis
    const pauses = this.detectPauses();
    const longPauses = pauses.filter(p => p.duration > this.longPauseThreshold).length;
    const avgPauseDuration = pauses.length > 0 ? 
      pauses.reduce((sum, p) => sum + p.duration, 0) / pauses.length : 0;
    const pauseFrequency = (pauses.length / sessionDuration) * 60;
    
    // Deletion analysis
    const deleteRatio = this.events.length > 0 ? 
      deleteEvents.length / this.events.length : 0;
    
    // Rhythm analysis
    const { burstiness, rhythmVariance } = this.analyzeRhythm();
    
    // Advanced metrics
    const hesitationScore = this.calculateHesitation(pauseFrequency, deleteRatio, longPauses);
    const emotionalIntensity = this.calculateEmotionalIntensity(wpm, rhythmVariance, this.maxDeleteStreak);
    const cognitiveLoad = this.calculateCognitiveLoad(pauseFrequency, deleteRatio, avgPauseDuration);
    
    // Update store
    this.metrics.set({
      wpm: Math.round(wpm),
      charactersPerMinute: Math.round(cpm),
      averagePauseDuration: Math.round(avgPauseDuration),
      pauseFrequency: Number(pauseFrequency.toFixed(2)),
      longPauses,
      deleteRatio: Number(deleteRatio.toFixed(3)),
      backspaceStreak: this.maxDeleteStreak,
      burstiness: Number(burstiness.toFixed(3)),
      rhythmVariance: Number(rhythmVariance.toFixed(3)),
      hesitationScore: Number(hesitationScore.toFixed(3)),
      emotionalIntensity: Number(emotionalIntensity.toFixed(3)),
      cognitiveLoad: Number(cognitiveLoad.toFixed(3))
    });
    
    // Generate insights
    this.generateInsights();
  }

  private detectPauses(): Array<{ duration: number; position: number }> {
    const pauses: Array<{ duration: number; position: number }> = [];
    
    for (let i = 1; i < this.events.length; i++) {
      const timeDiff = this.events[i].timestamp - this.events[i - 1].timestamp;
      if (timeDiff > this.pauseThreshold) {
        pauses.push({
          duration: timeDiff,
          position: this.events[i].position
        });
      }
    }
    
    return pauses;
  }

  private analyzeRhythm(): { burstiness: number; rhythmVariance: number } {
    if (this.events.length < 20) {
      return { burstiness: 0, rhythmVariance: 0 };
    }
    
    // Calculate inter-keystroke intervals
    const intervals: number[] = [];
    for (let i = 1; i < this.events.length; i++) {
      if (this.events[i].key !== 'PAUSE' && this.events[i - 1].key !== 'PAUSE') {
        intervals.push(this.events[i].timestamp - this.events[i - 1].timestamp);
      }
    }
    
    if (intervals.length < 10) {
      return { burstiness: 0, rhythmVariance: 0 };
    }
    
    // Remove outliers (pauses)
    intervals.sort((a, b) => a - b);
    const q1 = intervals[Math.floor(intervals.length * 0.25)];
    const q3 = intervals[Math.floor(intervals.length * 0.75)];
    const iqr = q3 - q1;
    const filtered = intervals.filter(i => i >= q1 - 1.5 * iqr && i <= q3 + 1.5 * iqr);
    
    // Calculate variance
    const mean = filtered.reduce((sum, i) => sum + i, 0) / filtered.length;
    const variance = filtered.reduce((sum, i) => sum + Math.pow(i - mean, 2), 0) / filtered.length;
    const stdDev = Math.sqrt(variance);
    
    // Burstiness: coefficient of variation
    const burstiness = mean > 0 ? stdDev / mean : 0;
    
    // Rhythm variance: normalized standard deviation
    const rhythmVariance = Math.min(stdDev / 1000, 1); // Normalize to 0-1
    
    return { burstiness: Math.min(burstiness, 1), rhythmVariance };
  }

  private calculateHesitation(pauseFreq: number, deleteRatio: number, longPauses: number): number {
    // High pause frequency, high delete ratio, and many long pauses indicate hesitation
    const pauseScore = Math.min(pauseFreq / 10, 1); // Normalize to 0-1
    const deleteScore = Math.min(deleteRatio * 5, 1); // Amplify delete impact
    const longPauseScore = Math.min(longPauses / 5, 1); // 5+ long pauses = max score
    
    return (pauseScore * 0.4 + deleteScore * 0.4 + longPauseScore * 0.2);
  }

  private calculateEmotionalIntensity(wpm: number, rhythmVar: number, maxDeletes: number): number {
    // Fast typing with high rhythm variance and deletion streaks indicate emotional intensity
    const speedScore = Math.min(wpm / 100, 1); // 100+ WPM = intense
    const rhythmScore = rhythmVar; // Already 0-1
    const deleteScore = Math.min(maxDeletes / 10, 1); // 10+ deletes in a row = intense
    
    return (speedScore * 0.5 + rhythmScore * 0.3 + deleteScore * 0.2);
  }

  private calculateCognitiveLoad(pauseFreq: number, deleteRatio: number, avgPause: number): number {
    // Frequent pauses, high deletion, and long average pauses indicate cognitive struggle
    const pauseFreqScore = Math.min(pauseFreq / 15, 1); // 15+ pauses/min = high load
    const deleteScore = Math.min(deleteRatio * 4, 1);
    const pauseDurationScore = Math.min(avgPause / 3000, 1); // 3+ second avg = high load
    
    return (pauseFreqScore * 0.3 + deleteScore * 0.4 + pauseDurationScore * 0.3);
  }

  private generateInsights(): void {
    const currentMetrics = this.getMetricsSnapshot();
    const insights: ProsodyInsight[] = [];
    
    // Emotional state insights
    if (currentMetrics.emotionalIntensity > 0.7) {
      insights.push({
        type: 'emotion',
        label: 'High Intensity',
        confidence: currentMetrics.emotionalIntensity,
        description: 'User appears emotionally engaged or excited'
      });
    } else if (currentMetrics.emotionalIntensity < 0.3) {
      insights.push({
        type: 'emotion',
        label: 'Calm',
        confidence: 1 - currentMetrics.emotionalIntensity,
        description: 'User appears calm and measured'
      });
    }
    
    // Confidence insights
    if (currentMetrics.hesitationScore > 0.6) {
      insights.push({
        type: 'confidence',
        label: 'Uncertain',
        confidence: currentMetrics.hesitationScore,
        description: 'User shows signs of uncertainty or careful consideration'
      });
    } else if (currentMetrics.hesitationScore < 0.2) {
      insights.push({
        type: 'confidence',
        label: 'Confident',
        confidence: 1 - currentMetrics.hesitationScore,
        description: 'User appears confident and decisive'
      });
    }
    
    // Cognitive load insights
    if (currentMetrics.cognitiveLoad > 0.7) {
      insights.push({
        type: 'cognitive',
        label: 'High Load',
        confidence: currentMetrics.cognitiveLoad,
        description: 'User may be struggling with complex thoughts'
      });
    }
    
    // Intent clarity
    if (currentMetrics.deleteRatio > 0.2 && currentMetrics.hesitationScore > 0.5) {
      insights.push({
        type: 'intent',
        label: 'Reformulating',
        confidence: 0.8,
        description: 'User is actively refining their message'
      });
    }
    
    this.insights.set(insights);
  }

  getMetricsSnapshot(): ProsodyMetrics {
    let snapshot: ProsodyMetrics = this.getEmptyMetrics();
    this.metrics.subscribe(m => snapshot = m)();
    return snapshot;
  }

  reset(): void {
    this.events = [];
    this.sessionStart = Date.now();
    this.lastEventTime = Date.now();
    this.deleteStreak = 0;
    this.maxDeleteStreak = 0;
    this.metrics.set(this.getEmptyMetrics());
    this.insights.set([]);
  }

  exportSession(): {
    metrics: ProsodyMetrics;
    insights: ProsodyInsight[];
    duration: number;
    eventCount: number;
  } {
    return {
      metrics: this.getMetricsSnapshot(),
      insights: (() => {
        let insights: ProsodyInsight[] = [];
        this.insights.subscribe(i => insights = i)();
        return insights;
      })(),
      duration: Date.now() - this.sessionStart,
      eventCount: this.events.length
    };
  }
}

// Export singleton instance
export const prosodyAnalyzer = new TypingProsodyAnalyzer();

// Export types
// Types exported individually above
