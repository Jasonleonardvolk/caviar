#!/bin/bash
# Git Commands for WebSocket Audio Streaming Feature

# =============================================================================
# SETUP FEATURE BRANCH
# =============================================================================

# 1. Ensure you're on main/master and it's up to date
git checkout main
git pull origin main

# 2. Create and switch to feature branch
git checkout -b feature/ws-audio-streaming

# =============================================================================
# ADD ALL MODIFIED FILES
# =============================================================================

# Backend Changes
echo "Adding backend files..."
git add ingest-bus/audio/worker.py
git add ingest-bus/audio/__init__.py  # If modified
git add tori_backend/routes/ws_audio.py
git add tori_backend/routes/ingest.py
git add tori_backend/routes/schemas.py
git add tori_backend/main.py

# Frontend Changes
echo "Adding frontend files..."
git add svelte/src/lib/stores/audio.ts
git add svelte/src/lib/components/AudioRecorder.svelte
git add svelte/src/lib/audioWorklet.ts

# Test Files
echo "Adding test files..."
git add tests/test_websocket.py
git add test_integration.py
git add QA_CHECKLIST.md

# Configuration Files (if any)
git add requirements.txt  # If updated
git add package.json      # If updated
git add .env.example      # If created

# =============================================================================
# CHECK STATUS
# =============================================================================

# Review what's being committed
git status

# Review changes in detail
git diff --cached

# =============================================================================
# COMMIT WITH DETAILED MESSAGE
# =============================================================================

# Commit with comprehensive message
git commit -m "feat: Add WebSocket audio streaming with real-time transcription

## Summary
Implement real-time audio streaming over WebSocket with live transcription,
spectral analysis, emotion detection, and holographic visualization hints.

## Backend Changes
- Implement AudioStreamProcessor with direct sample counting and buffering
- Add WebSocket route with non-blocking task management and queue system  
- Enhanced worker.py with overlap handling and performance metrics
- Update schemas with HologramHint model and cleaned AudioIngestionResult
- Add comprehensive error handling and activity monitoring

## Frontend Changes  
- Create reactive Svelte stores for live audio state management
- Enhance AudioRecorder component with streaming support
- Add AudioWorklet implementation with ScriptProcessor fallback
- Implement real-time UI updates for transcript, spectral, and emotion data
- Add hologram preview visualization

## Testing
- Add comprehensive WebSocket test suite with 15+ test cases
- Create integration test script with performance benchmarks
- Include QA checklist for manual verification
- Test concurrent connections, error recovery, and edge cases

## Features
- Real-time audio streaming with <500ms latency
- Live transcription with incremental updates
- Spectral analysis (centroid, RMS, spread, flux)
- Emotion detection with confidence scores
- Holographic visualization hints (hue, intensity, psi)
- Automatic reconnection and error recovery
- Performance monitoring and statistics

Closes #XXX"

# =============================================================================
# PUSH BRANCH
# =============================================================================

# Push branch to remote
git push -u origin feature/ws-audio-streaming

# =============================================================================
# CREATE PULL REQUEST
# =============================================================================

echo "
Next steps:
1. Go to GitHub/GitLab/Bitbucket
2. Create Pull Request from feature/ws-audio-streaming to main
3. Add reviewers
4. Link related issues
"

# =============================================================================
# ALTERNATIVE: INTERACTIVE STAGING
# =============================================================================

# If you want to review changes file by file:
# git add -i

# Or use patch mode to stage specific chunks:
# git add -p

# =============================================================================
# USEFUL COMMANDS
# =============================================================================

# View commit history
git log --oneline --graph --decorate

# Compare with main branch
git diff main..feature/ws-audio-streaming

# See what files changed
git diff --name-only main..feature/ws-audio-streaming

# =============================================================================
# AFTER PR APPROVAL
# =============================================================================

# Update feature branch with latest main
git checkout main
git pull origin main
git checkout feature/ws-audio-streaming
git merge main

# Or rebase (cleaner history)
git rebase main

# Push updates
git push origin feature/ws-audio-streaming

# =============================================================================
# CLEANUP AFTER MERGE
# =============================================================================

# After PR is merged, clean up local branch
git checkout main
git pull origin main
git branch -d feature/ws-audio-streaming

# Delete remote branch (if not auto-deleted)
git push origin --delete feature/ws-audio-streaming

# =============================================================================
# TAGS FOR RELEASE
# =============================================================================

# Tag the release
git tag -a v1.1.0 -m "Release v1.1.0: WebSocket audio streaming"
git push origin v1.1.0