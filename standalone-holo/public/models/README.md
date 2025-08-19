# ONNX Models Directory

This directory should contain the following ONNX models:

## Required Models:

### 1. depth_estimator.onnx
- **Purpose**: Monocular depth estimation from RGB images
- **Expected size**: 66-100 MB
- **Recommended model**: MiDaS v2.1 Small (256x256)
- **Download URL**: https://huggingface.co/julienkay/sentis-MiDaS/blob/main/midas_v21_small_256.onnx

To download automatically, run one of these from the standalone-holo directory:
```powershell
# PowerShell:
.\download-models.ps1

# Or Command Prompt:
download-models.bat
```

### 2. waveop_fno_v1.onnx (Optional)
- **Purpose**: Penrose neural wave operator for holographic field prediction
- **Status**: Optional - app will use deterministic fallback if missing
- **Note**: This would be your custom-trained FNO model

## Model Input/Output Specs:

### depth_estimator.onnx
- **Input**: NCHW tensor [1, 3, 256, 256] - RGB image
- **Output**: [1, 1, 256, 256] - Depth map

### waveop_fno_v1.onnx (when available)
- **Input**: [1, 1, H, W] - Depth map
- **Output**: Two tensors for amplitude and phase

## Troubleshooting:

If models fail to load:
1. Check file size (should be 60+ MB for depth model)
2. Ensure .onnx extension (not .onnx.download or similar)
3. Try the synthetic depth fallback by not placing any model
4. Check browser console for specific error messages

The app will automatically fall back to synthetic depth estimation if models are missing.