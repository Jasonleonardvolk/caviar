@echo off
echo ==========================================
echo 📥 DOWNLOAD DISCOVERED TIER 1 PAPERS 📥
echo ==========================================
echo.
echo 📚 Downloads all papers found by recursive mining
echo 🎯 Organizes by priority: Ultra-high, High, Medium, Citations
echo 📂 Saves to: C:\Users\jason\Desktop\tori\kha\data\tier1_consciousness_papers\
echo.
echo 📊 ORGANIZATION:
echo   📁 ultra_high_priority/    (Score 10.0+ ★★★★★)
echo   📁 high_priority/          (Score 5.0-9.9 ★★★★)  
echo   📁 medium_priority/        (Score 2.0-4.9 ★★★)
echo   📁 citation_discoveries/   (Found via citations)
echo.

REM Check if results file exists
if not exist "enhanced_recursive_tier1_results.json" (
    echo ❌ No results file found!
    echo 💡 Run the enhanced recursive miner first:
    echo    .\START_ENHANCED_RECURSIVE_TIER1_MINING.bat
    echo.
    pause
    exit /b 1
)

echo ✅ Results file found!
echo 📊 Checking discovered papers...

REM Show paper count from results
python -c "import json; data=json.load(open('enhanced_recursive_tier1_results.json')); print(f'📚 {len(data[\"papers\"])} Tier 1 papers ready to download!')"

echo.
echo 🚀 Starting download process...
echo ⏱️ Estimated time: 2-6 hours depending on paper count
echo 📥 Papers will be organized by priority level
echo.
echo 🍺 Another great beer time opportunity! 🍺

python download_tier1_papers.py

echo.
echo 🎉 Download complete!
echo 📂 Check: C:\Users\jason\Desktop\tori\kha\data\tier1_consciousness_papers\
echo 🔄 Next: Upload all papers to TORI with .\START_BEER_TIME_UPLOAD.bat
echo 🍺 Enjoy your consciousness paper library! 🍺
pause