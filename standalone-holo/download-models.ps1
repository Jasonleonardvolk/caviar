// download-models.ps1 - PowerShell script to download ONNX models
// Save this file and run it in PowerShell: .\download-models.ps1

Write-Host "Downloading depth estimation model..." -ForegroundColor Green

$modelPath = "C:\Users\jason\Desktop\tori\kha\standalone-holo\public\models\depth_estimator.onnx"

# Option 1: MiDaS v2.1 Small (66MB - recommended for speed)
$midasUrl = "https://huggingface.co/julienkay/sentis-MiDaS/resolve/b867253b86ef4cef1cfda70e8fbcf72fb27eaa3e/midas_v21_small_256.onnx?download=true"

# Option 2: Depth Anything v2 Small (99MB - better quality)
# Uncomment the line below to use this instead:
# $depthAnythingUrl = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx?download=true"

try {
    # Download MiDaS (change to $depthAnythingUrl if you prefer that model)
    Write-Host "Downloading from Hugging Face..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $midasUrl -OutFile $modelPath -UseBasicParsing
    
    $fileSize = (Get-Item $modelPath).length / 1MB
    Write-Host "✓ Model downloaded successfully! Size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Green
    Write-Host "Location: $modelPath" -ForegroundColor Cyan
    
} catch {
    Write-Host "✗ Download failed: $_" -ForegroundColor Red
    Write-Host "You can manually download from:" -ForegroundColor Yellow
    Write-Host $midasUrl
    Write-Host "And save to: $modelPath"
}

Write-Host "`nNote: WaveOp model (waveop_fno_v1.onnx) is optional - the app will use fallback if not present" -ForegroundColor Blue