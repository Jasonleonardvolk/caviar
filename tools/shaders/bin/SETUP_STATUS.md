# 🛠️ SHADER TOOLS BIN DIRECTORY CREATED

## ✅ Directory Created:
`${IRIS_ROOT}\tools\shaders\bin\`

## 📁 Files Created:
```
tools/shaders/bin/
├── README.md              ✅ Setup instructions
├── .gitkeep              ✅ Keep directory in git
├── .gitignore            ✅ Ignore .exe files
├── download_tint.ps1     ✅ Auto-download script
├── test_tint.cmd         ✅ Test installation
└── check_validators.mjs  ✅ Check available validators
```

## 📥 DOWNLOAD TINT.EXE:

### Option 1: Try Auto-Download
```powershell
cd ${IRIS_ROOT}\tools\shaders\bin
powershell -ExecutionPolicy Bypass -File download_tint.ps1
```

### Option 2: Manual Download

1. **Dawn/Tint Releases** (Recommended):
   - https://dawn.googlesource.com/dawn/
   - Look for: `tint.exe` or `tint-windows-amd64.exe`
   - Download and place in: `tools\shaders\bin\tint.exe`

2. **Chromium CI Builds**:
   - https://ci.chromium.org/p/dawn/builders/ci
   - Find Windows builds
   - Extract `tint.exe`

3. **Alternative - Naga** (Rust-based):
   ```bash
   # If you have Rust installed:
   cargo install naga-cli
   # Then copy naga.exe to tools\shaders\bin\
   ```

## 🧪 TEST INSTALLATION:

```cmd
cd ${IRIS_ROOT}\tools\shaders\bin
.\test_tint.cmd
```

Or check all validators:
```bash
node check_validators.mjs
```

## ✅ Success Criteria:
```
> tint.exe --version
tint version v0.0.1
```

## 🔗 Direct Download Links to Try:

- **Naga (alternative)**: https://github.com/gfx-rs/wgpu/releases/latest
  - Look for: `naga-cli-windows.zip`
  
- **Dawn Artifacts**: Search for "dawn tint windows binary" 
  - Often in CI artifacts or releases

## 📝 Notes:
- Tint.exe validates WGSL and converts to HLSL/MSL/SPIR-V
- Required for cross-platform shader validation
- Alternative: Naga from wgpu project also works
