@echo off
echo Creating branch and tag for WOW Pack + Social Pack release...

cd /d D:\Dev\kha

echo.
echo Creating release branch...
git checkout -b release/wowpack-socialpack

echo.
echo Adding all changes...
git add -A

echo.
echo Committing...
git commit -m "feat: WOW Pack (ProRes to HDR10/AV1/SDR) + Social Pack (Snap/TikTok 9:16) pipelines"

echo.
echo Pushing branch...
git push -u origin release/wowpack-socialpack

echo.
echo Creating tag...
git tag -a wowpack-socialpack-v1.0.0 -m "WOW v1 + Social v1: pipelines, verification, docs"

echo.
echo Pushing tag...
git push origin wowpack-socialpack-v1.0.0

echo.
echo ✅ DONE! Branch and tag created.
echo.
echo Branch: release/wowpack-socialpack  
echo Tag: wowpack-socialpack-v1.0.0
