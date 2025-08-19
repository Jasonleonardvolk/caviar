# Shader Tools Binaries

This directory contains binary tools for shader validation and compilation.

## Required Tools:

### tint.exe (WGSL Validator/Compiler)
- **Purpose**: Validates WGSL and compiles to HLSL/MSL/SPIR-V
- **Download**: https://github.com/google/dawn/releases
- **Version**: Latest stable (or v0.0.1 if available)

## Setup Instructions:

1. Download Tint from Dawn releases:
   - Go to: https://github.com/google/dawn/releases
   - Look for `tint-windows-amd64.exe` or similar
   - Or try: https://dawn.googlesource.com/dawn/

2. Alternative download locations:
   - ChromiumDev builds: https://chromium.googlesource.com/chromium/src/+/main/docs/testing/web_platform_tests.md
   - Direct from Dawn CI: https://ci.chromium.org/p/dawn/builders/ci

3. Place `tint.exe` in this directory:
   ```
   ${IRIS_ROOT}\tools\shaders\bin\tint.exe
   ```

4. Verify installation:
   ```bash
   cd ${IRIS_ROOT}\tools\shaders\bin
   .\tint.exe --version
   ```

## Expected Files:
- [ ] tint.exe - WGSL validator and cross-compiler
- [ ] naga.exe (optional) - Alternative WGSL validator
- [ ] dxc.exe (optional) - DirectX Shader Compiler for HLSL

## Path Setup (Optional):
Add this directory to your PATH for global access:
```powershell
$env:Path += ";${IRIS_ROOT}\tools\shaders\bin"
```
