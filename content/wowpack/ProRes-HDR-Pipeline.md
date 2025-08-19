# WOW Pack v1 - ProRes to HDR MP4 Pipeline

## Overview
Converts ProRes .mov masters to web-optimized MP4 variants with HDR10 support.

## Workflow

### 1. INPUT: ProRes Masters
Place your ProRes .mov files in `D:\Dev\kha\content\wowpack\input\`

**Required specs:**
- **Codec:** ProRes 422 HQ or ProRes 4444
- **Bit Depth:** 10-bit (yuv422p10le or yuv444p10le)
- **Color Space:** Rec.709 (will be mapped to BT.2020 PQ)
- **Resolution:** 1920×1080 minimum (2560×1440 or 4K preferred)
- **Background:** Black or transparent

**File names must be exactly:**
- `holo_flux_loop.mov`
- `mach_lightfield.mov`
- `kinetic_logo_parade.mov`

### 2. VERIFY: Check Your Masters
```powershell
cd D:\Dev\kha\tools\encode
.\Check-ProRes-Masters.ps1
```

This will verify:
- Files are present
- ProRes codec confirmed
- 10-bit color depth
- Proper color space

### 3. ENCODE: ProRes → MP4 Variants
```powershell
cd D:\Dev\kha\tools\encode
.\Batch-Encode-Simple.ps1
```

This creates for each video:
- **HEVC HDR10** (`*_hdr10.mp4`) - Hardware-accelerated HDR playback
- **AV1 10-bit** (`*_av1.mp4`) - Most efficient codec where supported
- **H.264 SDR** (`*_sdr.mp4`) - Universal fallback

### 4. VERIFY: Check Encoded Assets
```powershell
cd D:\Dev\kha\tools\release
.\Verify-WowPack.ps1
```

### 5. TEST: View in Browser
```
http://localhost:3000/hologram?clip=holo_flux_loop
http://localhost:3000/hologram?clip=mach_lightfield
http://localhost:3000/hologram?clip=kinetic_logo_parade
```

## File Locations

### Masters (ProRes .mov)
```
content\wowpack\input\
├── holo_flux_loop.mov
├── mach_lightfield.mov
└── kinetic_logo_parade.mov
```

### Archives (Encoded MP4s)
```
content\wowpack\video\
├── hdr10\
│   ├── *_hdr10.mp4  (HEVC HDR10)
│   └── *_sdr.mp4    (H.264 SDR)
└── av1\
    └── *_av1.mp4    (AV1 10-bit)
```

### Runtime (Web Serving)
```
tori_ui_svelte\static\media\wow\
├── wow.manifest.json
├── holo_flux_loop_hdr10.mp4
├── holo_flux_loop_av1.mp4
├── holo_flux_loop_sdr.mp4
└── [same for other clips...]
```

## Color Pipeline

### Input (ProRes)
- **Color Space:** Rec.709
- **Transfer:** BT.709 (gamma 2.4)
- **Bit Depth:** 10-bit

### HDR Output (HEVC/AV1)
- **Color Space:** BT.2020
- **Transfer:** SMPTE ST 2084 (PQ)
- **Bit Depth:** 10-bit
- **Max Luminance:** 1000 nits
- **HDR10 Metadata:** Embedded

### SDR Output (H.264)
- **Color Space:** Rec.709
- **Transfer:** BT.709
- **Bit Depth:** 8-bit
- **Tone Mapping:** Hable

## Encoding Parameters

### HEVC HDR10
```
-crf 18 (high quality)
-preset medium
-x265-params hdr10=1:hdr10-opt=1
-tag:v hvc1 (for Apple compatibility)
```

### AV1
```
-crf 30 (efficient)
-preset 6 (balanced speed/quality)
-svtav1-params enable-hdr=1
```

### H.264 SDR
```
-crf 18 (high quality)
-preset medium
-profile:v high -level 4.2
```

## Playback Selection (Capability-Based)

The system automatically detects hardware capabilities and selects the best format:

- **If hardware AV1 10-bit is efficient** → use `*_av1.mp4`
- **Else if HEVC 10-bit HDR10 is efficient** (hvc1/hev1) → use `*_hdr10.mp4`
- **Else** → use `*_sdr.mp4` (H.264)

No device model detection - pure capability-based selection.

## Troubleshooting

### "Not ProRes" Warning
Your .mov files must be ProRes for best quality. The encoder will fail if input is not ProRes 10-bit.

### HDR Not Showing
- Verify your display supports HDR
- Check that HEVC files have `color_transfer=smpte2084`
- Ensure True Tone is enabled on capable displays

### Files Too Large
- HEVC HDR10: ~50-100 MB expected
- AV1: ~30-50 MB expected
- SDR: ~30-60 MB expected

If larger, your source may be too high bitrate. Re-export from Premiere/AE with ProRes 422 HQ (not 4444 XQ).

## Quick Commands Reference

```powershell
# Check what you have
.\Check-ProRes-Masters.ps1

# Encode everything
.\Batch-Encode-Simple.ps1

# Verify results
..\release\Verify-WowPack.ps1

# Check a specific file's HDR metadata
ffprobe -v error -show_streams -select_streams v:0 "path\to\file_hdr10.mp4" | findstr "color"
```

## Export Settings (Premiere/After Effects)

### Premiere Pro
1. File → Export → Media
2. Format: QuickTime
3. Preset: ProRes 422 HQ
4. Color:
   - Working Color Space: Rec. 709
   - Bit Depth: 10-bit
5. Frame Rate: 60 fps (or native)

### After Effects
1. Composition → Add to Render Queue
2. Output Module:
   - Format: QuickTime
   - Format Options → Codec: ProRes 422 HQ
3. Color:
   - Depth: Millions of Colors+ (10-bit)
   - Profile: HD (1-1-1)

## Notes

- Always use `.mov` for masters (ProRes container)
- Always serve `.mp4` for web (browser compatible)
- The `hvc1` tag ensures iOS compatibility
- Manifests auto-update with each encoding
- HLS is optional (only for streaming needs)
