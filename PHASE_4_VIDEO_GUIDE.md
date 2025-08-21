# PHASE 4 - VIDEO CREATION GUIDE
# iRis Marketing Videos

## Video Specifications

### Video A: "Shock Proof" (10 seconds)
**Purpose**: Show the free tier with watermark - creates urgency to upgrade
**Duration**: 10 seconds
**Plan**: Free
**Key Visual**: Watermark clearly visible

**Script**:
1. Open hologram studio
2. Select vibrant/neon preset for impact
3. Click "Record 10s" (Free plan)
4. Let auto-stop at 10s
5. Watermark visible in lower-right
6. Message: "Free tier gets you started, but..."

**File**: `D:\Dev\kha\site\showcase\A_shock_proof.mp4`

---

### Video B: "How-To Tutorial" (20-30 seconds)
**Purpose**: Show the upgrade flow and Plus features
**Duration**: 20-30 seconds
**Plan**: Plus
**Key Visual**: No watermark, longer recording

**Script**:
1. Start on pricing page (0-5s)
2. Click "Get Plus" → Stripe checkout (5-10s)
3. Return to studio showing "Plus" badge (10-15s)
4. Record high-quality hologram (15-25s)
5. Download - NO watermark (25-30s)
6. Message: "Professional quality, no limits"

**File**: `D:\Dev\kha\site\showcase\B_how_to_60s.mp4`

---

### Video C: "Buyers Clip" (15-20 seconds)
**Purpose**: Show professional deliverables
**Duration**: 15-20 seconds  
**Plan**: Plus/Pro
**Key Visual**: File explorer with exports

**Script**:
1. Open File Explorer (0-3s)
2. Navigate to `D:\Dev\kha\exports\video\` (3-8s)
3. Show multiple MP4 files with pro names (8-12s)
4. Double-click to play one (12-17s)
5. End card: "Export to any format" (17-20s)

**File**: `D:\Dev\kha\site\showcase\C_buyers_clip.mp4`

---

## Recording Methods

### Method 1: In-App Recording (Recommended)
Use the built-in HologramRecorder:
- Already set up with correct settings
- Automatic watermark on Free tier
- Direct WebM output

### Method 2: Screen Recording

#### Windows Game Bar
1. Press `Win + G`
2. Click Record (or `Win + Alt + R`)
3. Stop with `Win + Alt + R`
4. Find in `Videos\Captures`

#### OBS Studio (Professional)
1. Download from obsproject.com
2. Add Source → Display Capture
3. Settings: 1920x1080, 30fps, MP4
4. Better quality than Game Bar

#### iPhone Screen Recording
1. Settings → Control Center → Add Screen Recording
2. Open site in Safari
3. Control Center → Record
4. Stop → Save to Photos
5. AirDrop to PC

---

## Video Conversion

### WebM to MP4
```powershell
# Use the converter script
.\tools\exporters\webm-to-mp4.ps1 -In "iris_12345.webm" -Out "A_shock_proof.mp4"
```

### FFmpeg Manual Command
```bash
ffmpeg -i input.webm -c:v libx264 -pix_fmt yuv420p -movflags +faststart output.mp4
```

---

## File Organization

### Local Storage
```
D:\Dev\kha\
├── site\
│   └── showcase\
│       ├── A_shock_proof.mp4
│       ├── B_how_to_60s.mp4
│       └── C_buyers_clip.mp4
└── exports\
    └── video\
        └── (demo files for Video C)
```

### Cloud Backup
```
Google Drive\
└── iRis\
    └── Showcase\
        ├── A_shock_proof.mp4
        ├── B_how_to_60s.mp4
        └── C_buyers_clip.mp4
```

---

## Quick Commands

### Start Recording Session
```powershell
.\Create-Marketing-Videos.ps1
```

### Create Demo Files
```powershell
.\Create-Marketing-Videos.ps1 -CreateDummyFiles
```

### Convert Videos
```powershell
.\tools\exporters\webm-to-mp4.ps1 -In "recording.webm"
```

---

## Production Tips

### Visual Guidelines
- **Contrast**: High contrast for mobile viewing
- **Movement**: Smooth, deliberate movements
- **Focus**: Center important elements
- **Text**: Ensure watermark is clearly visible (Video A)

### Technical Specs
- **Resolution**: 1920x1080 preferred
- **Frame Rate**: 30fps minimum
- **Bitrate**: 6-8 Mbps for quality
- **Format**: MP4 with H.264 for compatibility

### Social Media Optimization
- **Instagram**: 1:1 square crop, under 60s
- **Twitter**: 16:9, under 2:20
- **LinkedIn**: 16:9, under 10 minutes
- **TikTok**: 9:16 vertical, under 60s

---

## Checklist

### Pre-Recording
- [ ] Dev server running
- [ ] Clean desktop/browser
- [ ] Test recording works
- [ ] Dummy export files created

### Recording
- [ ] Video A recorded (10s, watermark)
- [ ] Video B recorded (20-30s, no watermark)
- [ ] Video C recorded (15-20s, exports)

### Post-Production
- [ ] Convert WebM to MP4
- [ ] Copy to showcase folder
- [ ] Upload to Google Drive
- [ ] Test playback on mobile

### Distribution
- [ ] Upload to social media
- [ ] Add to website
- [ ] Share with team
- [ ] Monitor engagement

---

## Success Metrics

- Video A: Creates urgency (watermark visible)
- Video B: Shows value (smooth upgrade)
- Video C: Proves quality (professional exports)

Combined: Complete story from free → paid → delivery