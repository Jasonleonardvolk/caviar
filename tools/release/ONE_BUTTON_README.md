# 🚀 IRIS One-Button Ship Script

## Complete Build & Package Script

This script handles everything needed to ship your IRIS project.

## Usage

### Basic (Full Build)
```powershell
.\tools\release\IrisOneButton.ps1
```

### Quick Build (Skip Checks)
```powershell
.\tools\release\IrisOneButton.ps1 -QuickBuild
```

### Custom Options
```powershell
# Skip TypeScript checking
.\tools\release\IrisOneButton.ps1 -SkipTypeCheck

# Skip shader validation
.\tools\release\IrisOneButton.ps1 -SkipShaderCheck

# Skip everything except build (fastest)
.\tools\release\IrisOneButton.ps1 -QuickBuild -SkipTypeCheck -SkipShaderCheck
```

## What It Does

### Step 1: Dependencies
- Installs all required npm packages
- Ensures @webgpu/types, vite, svelte are present

### Step 2: TypeScript Validation
- Runs `tsc --noEmit` to check for errors
- Blocks if >50 errors (unless -QuickBuild)
- Can skip with -SkipTypeCheck

### Step 3: Shader Validation ⭐ NEW!
- Runs shader gate against device limits
- Tests for iPhone 11 & iPhone 15 compatibility
- Uses Naga validator (+ Tint if available)
- Generates reports in tools\shaders\reports\
- Can skip with -SkipShaderCheck

### Step 4: Build
- Full build: `npm run build`
- Quick build: `npx vite build` (bypasses TypeScript)

### Step 5: Package
- Creates timestamped release in releases\iris_v1_[timestamp]\
- Includes build manifest with metadata
- Copies all dist files

### Step 6: Report
- Shows build summary
- Total duration
- Validation status
- Package location
- Option to open release folder

## Shader Gate Details

The shader validation step (`run_shader_gate.ps1`) provides:

- **Device-specific validation**: Tests against real device limits
- **Multi-target support**: Currently iPhone 11 & 15, easily extensible
- **Tint integration**: Cross-validates to MSL/HLSL/SPIR-V if tint.exe present
- **Report generation**: JSON, JUnit XML, and text summaries
- **Strict mode**: Fails on any shader errors (warnings allowed)

### Adding Device Targets

To add more devices, just extend the `-Targets` array:
```powershell
-Targets @("iphone11","iphone15","android_adreno","desktop_nvidia")
```

Ensure corresponding limit files exist in:
```
tools\shaders\device_limits\
  iphone11.json
  iphone15.json
  android_adreno.json  # Add this
  desktop_nvidia.json  # Add this
```

## Exit Codes

- 0: Success - ready to ship!
- 1: Dependency installation failed
- 2: Shader validation failed
- 3: Build failed

## Examples

### Production Build (Full Validation)
```powershell
.\tools\release\IrisOneButton.ps1
```

### Development Build (Fast)
```powershell
.\tools\release\IrisOneButton.ps1 -QuickBuild -SkipShaderCheck
```

### CI/CD Build
```powershell
.\tools\release\IrisOneButton.ps1 -Strict
if ($LASTEXITCODE -ne 0) { 
    throw "Build failed with code $LASTEXITCODE" 
}
```

## Troubleshooting

### TypeScript Errors
- Run without -SkipTypeCheck to see errors
- Use -QuickBuild to bypass if non-critical

### Shader Errors
- Check reports in tools\shaders\reports\
- Review device limit files in tools\shaders\device_limits\
- Ensure shaders are in frontend\lib\webgpu\shaders\

### Build Failures
- Check npm install succeeded
- Verify all dependencies in package.json
- Try -QuickBuild for direct Vite build

## Ship It! 🚀

When the script completes successfully, your package is ready in:
```
releases\iris_v1_[timestamp]\
```

Deploy this folder to your server/CDN and you're live!
