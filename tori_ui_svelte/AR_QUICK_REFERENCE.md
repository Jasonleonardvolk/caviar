# AR Assets Quick Reference Card

## 🎯 Status Check
```batch
.\ar-status-check.bat
```

## 📥 Download Models
| Model | Source | Place In |
|-------|--------|----------|
| Sneaker | [CGTrader](https://www.cgtrader.com/free-3d-models/sports/equipment/sneaker) | `assets\3d\luxury\sneakers\free\cgtrader_generic_sneaker\raw\` |
| Watch | [Sketchfab](https://sketchfab.com/3d-models/low-poly-watch-85e3fe10d5e9411fbcd409f7eab6524f) | `assets\3d\luxury\watches\free\sketchfab_lowpoly_watch\raw\` |
| Perfume | [Sketchfab](https://sketchfab.com/3d-models/3d-perfume-bottle-for-product-visualizations-6e58cd19fb4f49829411c7f97df14f6e) | `assets\3d\luxury\perfume\free\sketchfab_perfume_bottle\raw\` |

## 🔧 Process Models
```batch
# Check what needs processing
.\process-ar-models.bat check

# Process all available models
.\process-ar-models.bat
```

## ✅ Test System
```batch
# Full system test
.\test-ar-assets.bat

# Quick validator test
node test-validator-quick.mjs

# Test specific asset
node tools\assets\validate-manifest.mjs assets\3d\luxury\ASSET_MANIFEST.json
```

## 🚀 Run Application
```batch
# Start dev server
pnpm dev

# View AR assets
# http://localhost:3000/assets

# API endpoint
# http://localhost:3000/api/assets
```

## 📊 Key Limits
- **Max Triangles**: 100,000
- **Target FPS**: 30-60 on iPhone 11
- **Compression**: 70-90% reduction
- **Formats**: GLTF/GLB only

## 🛡️ Git Protection
Pre-push hook automatically validates:
- Manifest exists and is valid
- Triangle counts within limits
- All routes accessible
- API endpoints working

## 📁 Project Structure
```
tori_ui_svelte/
├── assets/3d/luxury/          # 3D models
├── src/routes/assets/         # Viewer UI
├── src/routes/api/assets/     # API endpoint
├── tools/assets/              # Processing tools
└── *.bat                      # Helper scripts
```

## ⚡ Commands Cheat Sheet
| Task | Command |
|------|---------|
| Status check | `.\ar-status-check.bat` |
| Process models | `.\process-ar-models.bat` |
| Test system | `.\test-ar-assets.bat` |
| Validate manifest | `pnpm run validate:assets` |
| Start dev | `pnpm dev` |
| Build prod | `pnpm build` |

## 🆘 Troubleshooting
| Issue | Solution |
|-------|----------|
| Models not found | Download from sources above |
| Validation fails | Check triangle count < 100k |
| Can't process | Run `.\tools\assets\install-dependencies.bat` |
| AR not working | Ensure HTTPS or use ngrok for mobile |

---
**System Status**: ✅ READY (awaiting model downloads)  
**Version**: 2.0  
**Last Updated**: January 2025