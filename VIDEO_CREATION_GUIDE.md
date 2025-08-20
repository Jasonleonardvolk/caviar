# 📹 iRis Marketing Video Creation Guide

## Prerequisites
1. Make sure your dev server is running:
   ```powershell
   cd D:\Dev\kha\tori_ui_svelte
   pnpm dev
   ```

2. Open your app in browser: http://localhost:5173/hologram

## Video A: Shock/Proof (30s)
**Message:** "You can't mix two light fields. We made it look like you can."

### Recording Steps:
1. **Set up for "before" shot (0-7s)**
   - Go to http://localhost:5173/hologram
   - Select "Ghost Fade" preset (looks plain/flat)
   - Click "Record 10s" (FREE plan - watermark visible)
   - Let it record for 7 seconds showing the basic effect
   - Stop recording
   - Save as: `D:\Dev\kha\site\showcase\A_before_raw.webm`

2. **Set up for "after" shot (7-14s)**
   - Refresh the page
   - Select "Neon" preset (high contrast, vibrant)
   - Simulate Plus mode: Open browser console (F12) and run:
     ```javascript
     localStorage.setItem('iris.plan', 'plus');
     location.reload();
     ```
   - Click "Record 60s" (no watermark)
   - Record 7-10 seconds of the dramatic effect
   - Stop recording
   - Save as: `D:\Dev\kha\site\showcase\A_after_pro.webm`

3. **Convert to MP4:**
   ```powershell
   cd D:\Dev\kha\tools\exporters
   .\webm-to-mp4.ps1 -In "D:\Dev\kha\site\showcase\A_before_raw.webm"
   .\webm-to-mp4.ps1 -In "D:\Dev\kha\site\showcase\A_after_pro.webm"
   ```

4. **Edit together (using Windows Photos app or any editor):**
   - Import both clips
   - 0-2s: Title card "You can't mix two light fields"
   - 2-7s: A_before_raw.mp4 (label: "Reality")
   - 7-14s: A_after_pro.mp4 (label: "iRis Look")
   - 14-21s: Split screen showing both
   - 21-27s: Text cards "Record 10s → Pick Look → Export"
   - 27-30s: End card "Free 10s export today. Founders-100: 50% off Pro"
   - Export as: `D:\Dev\kha\site\showcase\A_shock_proof.mp4`

## Video B: How-To in 60 Seconds (30s)
**Message:** "Make a sponsor-ready 'hologram' in 60 seconds"

### Recording Steps:
1. **Screen record the workflow:**
   - Since you can't screen record directly, simulate it:
   
2. **Part 1: Pick a preset (3s)**
   - Go to http://localhost:5173/hologram
   - Record yourself clicking between presets
   - Save as: `B_pick_preset.webm`

3. **Part 2: Recording process (7s)**
   - Show clicking "Record 60s" button
   - Let it record showing the countdown
   - Stop after 7 seconds
   - Save as: `B_recording.webm`

4. **Part 3: Export demo (7s)**
   - After recording stops, show the download happening
   - Open File Explorer to `D:\Dev\kha\exports\video\`
   - Show the files there
   - Save as: `B_export.webm`

5. **Convert all to MP4 and edit:**
   - Convert all webm files to mp4
   - Combine with text overlays:
     - 0-3s: "1) Pick a look"
     - 3-10s: "2) Record"
     - 10-17s: "3) Export (you own it)"
     - 17-24s: Show posting to social
     - 24-30s: "Free 10s export. Pro unlocks 60s + GLB/KTX2"

## Video C: The Buyer Clip (28-32s)
**Message:** "Brands don't buy filters. They buy deliverables."

### Recording Steps:
1. **Create a "deliverables" showcase:**
   - Open File Explorer
   - Navigate to `D:\Dev\kha\exports\`
   - Create some dummy files to show:
     ```powershell
     cd D:\Dev\kha\exports\video
     echo "dummy" > brand_campaign_001.mp4
     echo "dummy" > brand_campaign_001.mov
     echo "dummy" > brand_campaign_001_prores.mov
     cd ..\models
     echo "dummy" > hologram_asset.glb
     cd ..\textures_ktx2
     echo "dummy" > texture_001.ktx2
     ```

2. **Record the file explorer (15s):**
   - Use Windows Game Bar (Win+G) to record just the File Explorer window
   - Pan through the folders showing the different export formats
   - Save as: `C_deliverables.mp4`

3. **Record side-by-side comparison (10s):**
   - Go back to hologram page
   - Record split between basic and pro look
   - Add text: "Sponsor slate ready"
   - Save as: `C_comparison.webm`

4. **Final edit:**
   - 0-5s: Show deliverables grid
   - 5-12s: Side-by-side comparison
   - 12-20s: Mock case "48h turnaround • 3 looks • ProRes HQ"
   - 20-28s: "Featured on our Showcase" grid
   - 28-32s: "Creators: #HologramDrop today"

## Quick Alternative: Using Windows Game Bar
If the in-app recorder isn't working perfectly:

1. **Enable Game Bar:**
   - Press `Win+G` to open Game Bar
   - Click the record button (or press `Win+Alt+R`)
   - Record your browser window showing the hologram

2. **Record each segment:**
   - Record 10-15 second clips of each preset
   - Save to `D:\Dev\kha\site\showcase\`

## Post-Recording Checklist
- [ ] All videos saved to `D:\Dev\kha\site\showcase\`
- [ ] Converted to MP4 format
- [ ] Each video is 25-34 seconds
- [ ] Watermark visible in FREE demo
- [ ] No watermark in PLUS/PRO demos
- [ ] File paths visible in "deliverables" video
- [ ] Clear CTA at end of each video

## Upload Locations
1. **X/Twitter:** Upload directly with tweet
2. **TikTok:** Use mobile app, add trending music
3. **Instagram Reels:** Upload via mobile, add captions
4. **LinkedIn:** Upload natively, tag as "Product Launch"
5. **YouTube Shorts:** Upload as Shorts (vertical format)

## Captions to Copy/Paste

### Video A Caption:
```
Real talk: you can't physically merge display photons with the real world without optics. We faked the look—fast. Record → export → own the master. Founders-100 (50% off Pro) live today. #HologramDrop
```

### Video B Caption:
```
3 steps. 60 seconds. Sponsor-ready clips you actually own. Templates ship weekly. Free 10s export today—upgrade when it pays off.
```

### Video C Caption:
```
Ownable assets. Fast turnaround. Template cadence weekly. If you're a buyer: DM 'STUDIO'. Creators: post with #HologramDrop to get featured and win 1-yr Pro.
```
