@echo off
echo PROSODY ENGINE TEST
echo ===================
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

echo Testing prosody engine...
poetry run python -c "from prosody_engine.core import NetflixKillerProsodyEngine; print('SUCCESS: Engine imported!'); engine = NetflixKillerProsodyEngine(); print(f'SUCCESS: {len(engine.emotion_categories)} emotions loaded!')"

echo.
pause