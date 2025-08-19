@echo off
echo ==========================================
echo 🍺 ULTIMATE BEER TIME AUTOMATION 🍺  
echo ==========================================
echo.
echo 🤖 DOUBLE BEAST MODE ACTIVATED!
echo.
echo PHASE 1: Download fresh papers from arXiv
echo PHASE 2: Upload EVERYTHING (USB + fresh downloads)
echo.
echo ⏱️ Total time: 4-8 hours (perfect for a brewery tour!)
echo 📚 Result: MASSIVE knowledge foundation
echo.

echo 🚀 Starting Phase 1: Fresh arXiv downloads...
call START_ARXIV_DOWNLOAD.bat

echo.
echo 🔄 Phase 1 complete! Starting Phase 2: Upload everything...
echo.
echo 📤 Now uploading:
echo   • Your 23GB USB drive papers 
echo   • Fresh arXiv downloads
echo   • All will be processed automatically!
echo.

REM Update the upload script to include both directories
python -c "
import json
config = {
    'directories': [
        r'C:\Users\jason\Desktop\tori\kha\data\USB Drive',
        r'C:\Users\jason\Desktop\tori\kha\data\arxiv_downloads'
    ],
    'note': 'Processing both USB drive and fresh arXiv downloads'
}
with open('multi_dir_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('✅ Multi-directory config created')
"

call START_BEER_TIME_UPLOAD.bat

echo.
echo 🎉 ULTIMATE AUTOMATION COMPLETE!
echo 📚 Your knowledge base is now LEGENDARY!
echo 🧠 Thousands of papers processed and ready
echo 🍺 Hope you had an amazing beer adventure! 🍺
echo.
echo 🔮 Next: Start building your multi-tenant empire!
pause