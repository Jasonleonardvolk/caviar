# TORI Holographic System - QA Checklist

## Prerequisites
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Redis server running (for caching)
- [ ] Chrome/Firefox with WebRTC support
- [ ] Microphone permissions enabled

## 1. Backend Setup & Verification

### Start Backend Server
```bash
cd tori_backend
uvicorn main:app --reload --log-level info
```

### Verify Backend Health
- [ ] Navigate to http://localhost:8000/
  - Should show: `{"message":"TORI Holographic API","version":"1.0.0"}`
- [ ] Check health endpoint: http://localhost:8000/health
  - Status should be "healthy"
  - CPU/Memory/Disk usage should be reasonable
- [ ] Check API docs: http://localhost:8000/docs
  - All endpoints should be listed
  - WebSocket endpoints visible

## 2. Integration Tests

### Run Automated Integration Tests
```bash
python test_integration.py
```

### Expected Results:
- [ ] ✅ Server Health Check - All endpoints responding
- [ ] ✅ WebSocket Connection - Connected with correct config
- [ ] ✅ Audio Streaming - All audio types processed
- [ ] ✅ Concurrent Connections - 3/3 successful
- [ ] ✅ Error Handling - Graceful recovery
- [ ] ✅ Performance - Throughput > 0.1 MB/s

## 3. Frontend Setup & Verification

### Start Frontend Development Server
```bash
cd svelte
npm install  # First time only
npm run dev
```

### Verify Frontend Loading
- [ ] Navigate to http://localhost:5173
- [ ] Page loads without console errors
- [ ] Audio Recorder component visible
- [ ] No CORS errors in console

## 4. Manual WebSocket Streaming Test

### Test Procedure:
1. **Open Developer Console** (F12)
   - [ ] No errors on page load
   - [ ] WebSocket connection established message

2. **Test Recording Flow**
   - [ ] Click "Record" button
   - [ ] Browser asks for microphone permission (first time)
   - [ ] Recording indicator shows (red dot or border)
   - [ ] Visualization canvas shows activity

3. **Live Streaming Verification** (While Recording)
   
   **Audio Level Meters:**
   - [ ] Green level meter responds to voice
   - [ ] Peak indicator holds maximum level
   - [ ] Clipping warning appears if too loud

   **Live Transcript:**
   - [ ] Text appears within 200-500ms of speaking
   - [ ] Transcript updates incrementally
   - [ ] No duplicate words/phrases

   **Spectral Analysis:**
   - [ ] Frequency value changes with pitch
   - [ ] Higher pitch = higher frequency (Hz)
   - [ ] Silent periods show ~0 Hz

   **Emotion Detection:**
   - [ ] Emotion label updates (neutral/calm/excited/energetic)
   - [ ] Confidence percentage shown
   - [ ] Changes respond to voice tone

   **Hologram Preview:**
   - [ ] Color orb visible
   - [ ] Hue changes with frequency
   - [ ] Intensity changes with volume
   - [ ] Smooth transitions

4. **Stop Recording**
   - [ ] Click "Stop" button
   - [ ] Final transcript displayed
   - [ ] Processing indicator shown briefly
   - [ ] Download button appears

### Console Monitoring
Watch for these messages in browser console:
- [ ] "Audio WebSocket connected"
- [ ] "Connected with session: ..."
- [ ] No repeated error messages
- [ ] No "WebSocket not connected" errors

## 5. Feature-Specific Tests

### A. Visualization Modes
- [ ] Click waveform button - shows audio waveform
- [ ] Click spectrum button - shows frequency bars
- [ ] Click ψ (psi) button - shows oscillator visualization
- [ ] All visualizations animate smoothly

### B. Audio Quality Settings
- [ ] Change quality to Low - recording still works
- [ ] Change to High - better audio quality
- [ ] Settings persist between recordings

### C. Noise Control
- [ ] Toggle Noise Suppression - background noise reduced
- [ ] Toggle Echo Cancellation - no feedback loops
- [ ] Toggle Auto Gain - volume normalizes

### D. Streaming Performance
- [ ] Record for 30+ seconds continuously
- [ ] No significant lag buildup
- [ ] Memory usage stable (check Task Manager)
- [ ] No dropped chunks warnings

## 6. Error Handling Tests

### Network Interruption
1. Start recording
2. Disable network adapter briefly
3. Re-enable network
- [ ] Error message appears
- [ ] Reconnection attempted
- [ ] Can resume after reconnection

### Server Restart
1. Start recording
2. Stop backend server (Ctrl+C)
3. Restart backend server
- [ ] Frontend shows disconnection
- [ ] Auto-reconnects when server returns
- [ ] New session ID assigned

### Invalid Audio
- [ ] Mute microphone while recording
- [ ] Should show "silence" indicator
- [ ] No errors thrown

## 7. Performance Benchmarks

### Check WebSocket Status
```bash
curl http://localhost:8000/api/v1/ws/audio/status
```
- [ ] Active connections count correct
- [ ] No excessive memory usage

### Monitor Backend Logs
Look for:
- [ ] Processing times < 100ms per chunk
- [ ] No repeated errors
- [ ] Cleanup messages for closed connections

## 8. Unit Tests

### Run Backend Tests
```bash
cd tori_backend
pytest tests/ -v --cov
```
- [ ] All tests pass
- [ ] Coverage > 80%

### Run Specific Test Suites
```bash
# WebSocket tests
pytest tests/test_websocket.py -v

# Audio processing tests  
pytest tests/test_audio_processing.py -v

# API endpoint tests
pytest tests/test_api.py -v
```

## 9. Browser Compatibility

Test on multiple browsers:
- [ ] Chrome (latest) - Full functionality
- [ ] Firefox (latest) - Full functionality
- [ ] Safari - Basic functionality (may lack AudioWorklet)
- [ ] Edge - Full functionality

## 10. Mobile Testing (if applicable)

- [ ] Page loads on mobile browser
- [ ] Microphone permission works
- [ ] Touch interactions function
- [ ] Performance acceptable

## 11. Load Testing (Optional)

### Concurrent Users Test
```python
# Run multiple integration tests simultaneously
python test_integration.py &
python test_integration.py &
python test_integration.py &
```
- [ ] Server handles 3+ concurrent streams
- [ ] No significant performance degradation

## 12. Final Verification

### Complete User Journey
1. [ ] Open fresh browser session
2. [ ] Navigate to app
3. [ ] Record 10-second audio clip
4. [ ] Verify all real-time features work
5. [ ] Stop recording
6. [ ] Download recording
7. [ ] Verify downloaded file plays correctly

### Check Logs
- [ ] No unhandled exceptions in backend
- [ ] No memory leaks after extended use
- [ ] WebSocket connections cleaned up properly

## Sign-off Checklist

- [ ] All automated tests pass
- [ ] Manual testing completed without blockers
- [ ] Performance meets requirements (<500ms latency)
- [ ] Error handling works correctly
- [ ] Documentation is current
- [ ] No critical console errors
- [ ] User experience is smooth

## Known Issues / Limitations

Document any discovered issues:
1. _________________________________
2. _________________________________
3. _________________________________

## Test Environment

- OS: _________________________________
- Browser: _____________________________
- Python Version: ______________________
- Node Version: ________________________
- Date Tested: _________________________
- Tested By: ___________________________

---

## Quick Smoke Test (5 minutes)

For rapid verification after changes:

1. Start backend: `uvicorn main:app --reload`
2. Run: `python test_integration.py`
3. Start frontend: `npm run dev`
4. Record 5 seconds of audio
5. Verify:
   - [ ] Live transcript appears
   - [ ] Hologram preview updates
   - [ ] No console errors
6. Stop and check final result

✅ **PASS** / ❌ **FAIL**

Notes: _________________________________