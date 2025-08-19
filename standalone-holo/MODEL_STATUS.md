# ONNX Model Setup Status

## ✅ Setup Complete!

Your app now has:
1. **Automatic fallback** - Works WITHOUT models (synthetic depth)
2. **Model downloader scripts** - Easy one-command download
3. **Status indicators** - Shows which models are loaded in the UI

## 🚀 Quick Start (Choose One):

### Option A: Run with Synthetic Depth (NOW)
```bash
cd ${IRIS_ROOT}\standalone-holo
npm install
npm run dev
```
- Works immediately
- No download needed
- Uses synthetic depth estimation

### Option B: Download Real Model, Then Run
```bash
cd ${IRIS_ROOT}\standalone-holo
npm install

# Download model (choose one):
npm run download:models        # Uses PowerShell
# OR
npm run download:models:cmd     # Uses Command Prompt
# OR
.\download-models.ps1          # Direct PowerShell
# OR
download-models.bat            # Direct batch file

npm run dev
```

## 📊 Model Status in UI

The app shows model status in bottom-right corner:
- ✓ Loaded = Real ONNX model active
- ✗ Using Synthetic/Fallback = No model, using synthetic

## 🧪 Testing

1. Click **"Start Demo"** - See parallax effect
2. Click **"Test Pattern"** - Process synthetic test image
3. Click **"Upload Image"** - Process real image
4. Toggle **"Parallax"** - Enable/disable motion tracking

## 📦 Model Details

### Downloaded Model (MiDaS v2.1 Small)
- Size: ~66 MB
- Input: 256x256 RGB
- Output: 256x256 depth map
- Performance: ~100ms inference on mobile GPU

### Synthetic Fallback
- Instant (no download)
- Based on luminance + radial gradient
- Good for testing pipeline
- Lower quality depth but works everywhere

## 🔧 Troubleshooting

If download fails:
1. Check internet connection
2. Try manual download from: https://huggingface.co/julienkay/sentis-MiDaS/blob/main/midas_v21_small_256.onnx
3. Save to: `public/models/depth_estimator.onnx`

If model doesn't load:
- Check browser console for errors
- Verify file is named exactly `depth_estimator.onnx`
- Try synthetic mode first to verify pipeline

## ✨ Next Steps

With models working:
1. Upload your own images to test depth
2. Adjust `kScale` in `field_from_depth.ts` for different phase effects
3. Add your trained `waveop_fno_v1.onnx` when ready

The system is READY TO RUN with or without models!