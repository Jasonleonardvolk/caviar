# WOW Pack v1 - Quick Start Guide

## Directory Structure Created
```
D:\Dev\kha\
  content\wowpack\
    input\              # Place your ProRes/HDR masters here (.mov, .mp4)
    video\
      hdr10\            # HEVC10 outputs (authoritative archive)
      av1\              # AV1 10-bit mirrors
    stills\hdr\         # HDR stills for shader comp
    grading\luts\       # Optional .cube LUTs
    
  tori_ui_svelte\
    src\lib\
      video\            # Video source selection logic
      overlays\presets\ # Visual effect presets
      show\             # Show controller and recipes
    static\media\
      wow\              # Runtime-served progressive MP4s
      hls\              # Optional HLS streams
      
  tools\
    encode\             # Build-WowPack.ps1
    release\            # Verify-WowPack.ps1
```

## Quick Start

### 1. Place Your Video Masters
Drop your ProRes/HDR video files into:
```
D:\Dev\kha\content\wowpack\input\
```

### 2. Encode Videos
Run the encoding script for each video:

```powershell
cd D:\Dev\kha\tools\encode

# Basic encoding (HEVC + AV1)
.\Build-WowPack.ps1 -Basename holo_flux_loop -Input "..\..\content\wowpack\input\holo_flux_loop.mov"

# With SDR fallback
.\Build-WowPack.ps1 -Basename mach_lightfield -Input "..\..\content\wowpack\input\mach_lightfield.mov" -DoSDR

# With HLS streaming
.\Build-WowPack.ps1 -Basename kinetic_logo_parade -Input "..\..\content\wowpack\input\kinetic_logo_parade.mov" -DoSDR -MakeHLS
```

### 3. Verify Assets
Check all videos are properly encoded:

```powershell
cd D:\Dev\kha\tools\release
.\Verify-WowPack.ps1
```

### 4. Test in Browser
Start your services and navigate to:

- **Shader mode**: http://localhost:3000/hologram
- **Video mode**: http://localhost:3000/hologram?clip=holo_flux_loop
- **Different clips**: 
  - ?clip=mach_lightfield
  - ?clip=kinetic_logo_parade

## Features

### Automatic Codec Selection
- **AV1**: Latest devices (A17/M-series)
- **HEVC HDR10**: iPhone 11-15, modern tablets
- **H.264 SDR**: Universal fallback

### Overlay Effects
- **EdgeGlow**: Sobel edge detection with bloom
- **FresnelRim**: Rim lighting effect
- **Bloom**: Threshold-based glow

### Show Recipes
Combine video clips with overlay effects and audio reactivity.

## Encoding Parameters

### HEVC HDR10
- 10-bit color (yuv420p10le)
- BT.2020 color space
- SMPTE ST 2084 (PQ) transfer
- CRF 18 (high quality)
- HDR10 metadata embedded

### AV1
- 10-bit color
- Preset 6 (balanced)
- CRF 30 (efficient)
- Hardware decode support check

### H.264 SDR (Optional)
- 8-bit color (yuv420p)
- BT.709 color space
- High profile, Level 4.2
- CRF 18 (high quality)

## Troubleshooting

### FFmpeg Not Found
Install FFmpeg and add to PATH:
```powershell
winget install ffmpeg
```

### Videos Not Playing
1. Check browser console for errors
2. Verify codec support with Verify-WowPack.ps1
3. Try SDR fallback with -DoSDR flag

### Manifest Not Updating
The manifest is automatically updated each time you run Build-WowPack.ps1

## Performance Tips

1. **Pre-encode all variants** before demos
2. **Use local SSD** for video storage
3. **Test on target devices** (iPhone, iPad)
4. **Monitor GPU usage** during playback

## Next Steps

1. Add more clips to build your library
2. Create custom overlay presets
3. Design show recipes for different moods
4. Implement crossfade transitions
5. Add audio reactive visualizations
