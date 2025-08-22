// TEMPORARY WORKAROUND for intentTracker error
// Create this as a new file: intentTrackerStores.js

import { intentTracker } from '$lib/services/intentTracking';
import { derived } from 'svelte/store';

// Create a derived store that wraps the entire intentTracker
export const intentTrackerStore = derived(
  [intentTracker.currentIntent, intentTracker.conversationInsights, intentTracker.intentHistory],
  ([$currentIntent, $conversationInsights, $intentHistory]) => ({
    currentIntent: $currentIntent,
    conversationInsights: $conversationInsights,
    intentHistory: $intentHistory,
    // Add any other properties you need
    isCoherent: $currentIntent ? $currentIntent.coherenceScore > 0.7 : true,
    needsClarification: $currentIntent ? $currentIntent.coherenceScore < 0.4 : false
  })
);

// Re-export the individual stores for direct use
export const { currentIntent, conversationInsights, intentHistory } = intentTracker;
