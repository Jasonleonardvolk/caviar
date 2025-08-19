# 🎯 TORI QUICK START GUIDE - BANKSY INTEGRATION

## ⚡ 30-SECOND LAUNCH

**TORI with Banksy is 100% ready for deployment!** Here's how to launch:

### Windows Users:
```cmd
# Double-click this file:
deploy-tori-production.bat
```

### Linux/Mac Users:
```bash
chmod +x deploy-tori-production.sh
./deploy-tori-production.sh
```

### Manual Launch:
```bash
# Terminal 1: Start Banksy Backend
cd alan_backend/server
python simulation_api.py

# Terminal 2: Start Frontend  
npm start
```

---

## 🌐 Access Points

Once launched, access TORI at:

- **🖥️ Main Interface**: http://localhost:3000
- **🌀 Banksy API**: http://localhost:8000  
- **📊 API Docs**: http://localhost:8000/docs
- **🔌 WebSocket**: ws://localhost:8000/ws/simulate

---

## 🎮 Using the Banksy Oscillator

1. **Locate Panel**: Look for 🌀 Banksy Oscillator panel (top-left)
2. **Check Status**: Green dot = Connected, Red = Disconnected
3. **Configure**: Set oscillator count (4-256) and coupling type
4. **Launch**: Click "Start Simulation"  
5. **Monitor**: Watch order parameter rise to 1.0 = full sync

### Configuration Options:
- **Oscillators**: 4, 8, 16, 32, 64, 128, 256
- **Coupling**: Modular (communities), Uniform (all-to-all), Random
- **Steps**: 10-1000 simulation steps
- **Substeps**: 1-16 spin integration steps

---

## 🔍 Verification Checklist

After launch, verify these indicators:

### UI Status (Top-Left Corner):
- ✅ **"Cognition Active"** - Green dot
- ✅ **"🌀 Banksy Online"** - Blue dot  

### Banksy Panel:
- ✅ **Status**: "🟢 Simulation Running" or "🟡 Ready"
- ✅ **Connection**: No red error messages
- ✅ **Metrics**: Order parameter, sync count updating

### Browser Console:
- ✅ **No WebSocket errors**
- ✅ **"🧠 TORI Cognition Engine: All systems operational"**
- ✅ **"🔗 Banksy WebSocket connected"**

---

## 🎯 Key Features Demonstration

### 1. Phase Synchronization
- Start with 32 oscillators, modular coupling
- Watch order parameter climb from 0.0 → 1.0
- Observe individual oscillator phases align on circle

### 2. Real-time Visualization  
- Phase circle shows oscillator positions
- Green arrow = order parameter vector
- Progress bar = synchronization percentage

### 3. Cognitive Integration
- High sync triggers holographic effects
- Phase coherence influences ghost persona emergence
- Agent field harmony responds to oscillator state

### 4. Performance Monitoring
- Debug panel shows FPS, memory usage
- TRS loss monitors temporal stability
- Connection status tracks backend health

---

## 🧠 Understanding the Integration

### What is Banksy?
The **Banksy Oscillator** is TORI's cognitive rhythm engine. It provides:
- **Phase synchronization** for coherent thought processes
- **Temporal stability** through TRS (time-reversal symmetry)
- **Memory consolidation** via spin-Hopfield networks
- **Agent coordination** through field harmonics

### How it Connects to TORI:
1. **Oscillator State** → **Phase Coherence** in cognition engine
2. **Order Parameter** → **Holographic Effect Intensity**  
3. **Synchronization** → **Ghost Persona Emergence**
4. **TRS Loss** → **Memory Stability Monitoring**

---

## 🔧 Troubleshooting

### Backend Won't Start:
```bash
# Check Python dependencies
pip install fastapi uvicorn websockets numpy

# Try different port
python simulation_api.py --port 8001
```

### Frontend Connection Issues:
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies  
rm -rf node_modules package-lock.json
npm install
```

### WebSocket Disconnections:
- Check firewall settings (allow ports 3000, 8000)
- Verify no antivirus blocking local connections
- Try restarting with `stop-tori-production.bat` then restart

### Performance Issues:
- Reduce oscillator count (try 16 instead of 64)
- Lower update frequency in debug panel
- Close unnecessary browser tabs

---

## 🎉 Success Indicators

**You'll know TORI+Banksy is working perfectly when:**

1. **🟢 All status dots are green** in the UI
2. **🌀 Banksy panel shows real-time metrics** updating every 100ms
3. **📊 Order parameter rises smoothly** from 0.0 to 0.8+ during sync
4. **🎨 Visual effects trigger** when synchronization reaches high levels
5. **⚡ No errors in browser console** or terminal windows

---

## 🚀 Next Steps

Once TORI+Banksy is running:

### Explore Cognitive Features:
- **📝 Chat Interface**: Test conversation with ghost personas
- **🧠 Memory System**: Upload documents, watch concept integration
- **🎛️ Agent Coordination**: Observe multi-agent field dynamics
- **📊 Analytics**: Monitor phase coherence and stability metrics

### Advanced Configuration:
- **🔧 Coupling Matrices**: Experiment with different network topologies
- **⏱️ Temporal Dynamics**: Adjust integration timesteps
- **🧮 Memory Patterns**: Load specific Hopfield attractors  
- **🎭 Persona Tuning**: Configure ghost emergence thresholds

### Development & Extension:
- **📈 Custom Visualizations**: Add new phase space displays
- **🔌 API Integration**: Connect external cognitive services
- **📚 Knowledge Import**: Bulk load domain-specific concepts
- **🎯 Reasoning Tasks**: Test complex multi-step problems

---

## 📞 Support

If you encounter issues:

1. **📋 Check Logs**: Look in `logs/` directory for error details
2. **🌐 API Docs**: Visit http://localhost:8000/docs for backend info  
3. **🔍 Debug Panel**: Enable in UI for real-time diagnostics
4. **🔄 Restart**: Use stop/start scripts for clean restart

**TORI with Banksy represents the cutting edge of cognitive architecture - enjoy exploring the synchronized intelligence!** 🧠✨
