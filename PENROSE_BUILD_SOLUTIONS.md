# Penrose Build Issue - Solutions

The Rust build failed, likely due to missing Visual Studio C++ Build Tools. Here are your options:

## Option 1: Quick Fix (Immediate)
**Run:** `PENROSE_QUICK_FIX.bat`
- Uses Python fallback
- Works immediately
- Slower but fully functional
- Good for testing TORI right now

## Option 2: Diagnose & Fix Rust Build (Best Performance)
**Step 1:** `DIAGNOSE_PENROSE_BUILD.bat`
- Shows exactly what's missing
- Usually Visual Studio Build Tools

**Step 2:** Install missing components
- If missing C++ tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
- Select "Desktop development with C++" workload
- Restart terminal after install

**Step 3:** `BUILD_PENROSE.bat`
- Should work after installing tools
- 80-100x faster than Python

## Option 3: Numba Acceleration (Good Compromise)
**Run:** `INSTALL_PENROSE_NUMBA.bat`
- No C++ toolchain needed
- 10-20x faster than pure Python
- Easy installation
- Good middle ground

## Performance Comparison
| Method | Speed | Setup Difficulty |
|--------|-------|------------------|
| Rust | 80-100x | Hard (needs C++) |
| Numba | 10-20x | Easy |
| Python | 1x (baseline) | None |

## Common Issues

### "error: Microsoft Visual C++ 14.0 is required"
- Install Visual Studio Build Tools
- Select C++ workload

### "linker `link.exe` not found"
- Same solution as above
- May need to restart terminal

### "cl.exe not found"
- C++ compiler not in PATH
- Install Build Tools fixes this

## Recommendation
1. Run `PENROSE_QUICK_FIX.bat` to get TORI running now
2. Run `DIAGNOSE_PENROSE_BUILD.bat` to see what's needed
3. Either:
   - Install Build Tools for best performance, OR
   - Use `INSTALL_PENROSE_NUMBA.bat` for easy setup with good performance