Param([string]$Profile = "desktop_low", [string]$Targets = "naga")
$limits = "tools/shaders/device_limits/$Profile.json"
node tools/shaders/validate_and_report.mjs --dir=frontend/lib/webgpu/shaders --limits=$limits --targets=$Targets --strict