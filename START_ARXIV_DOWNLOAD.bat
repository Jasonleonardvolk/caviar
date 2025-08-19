@echo off
echo ==========================================
echo 📚 ARXIV AUTO-DOWNLOADER BEAST 📚
echo ==========================================
echo.
echo 🧠 About to download FRESH papers from arXiv
echo 🎯 Categories: AI, ML, Physics, Biology, Finance, etc.
echo 🌟 Priority-based: Focuses on consciousness, AI, quantum topics
echo 🍺 Perfect beer time - this will take HOURS!
echo.

echo 🔍 Installing required packages...
pip install arxiv requests >nul 2>&1

echo ✅ Packages ready!
echo.
echo 🚀 Starting the download beast...
echo 📊 Will download up to 1000 papers (100 per category)
echo 📂 Saving to: C:\Users\jason\Desktop\tori\kha\data\arxiv_downloads
echo 🔄 Progress saved automatically - can resume if interrupted
echo.
echo 🍺 BEER TIME! Go relax while the beast hunts for knowledge!
echo.

python arxiv_downloader.py

echo.
echo 🎉 Download complete! 
echo 📚 Fresh arXiv papers ready for upload!
echo 🔄 Run the upload beast next: .\START_BEER_TIME_UPLOAD.bat
echo 🍺 Hope you enjoyed your beer! 🍺
pause