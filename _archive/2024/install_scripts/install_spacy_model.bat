@echo off
echo.
echo ========================================
echo Installing SpaCy Language Model
echo ========================================
echo.

echo Downloading English language model for spaCy...
python -m spacy download en_core_web_sm

echo.
echo Testing spaCy model...
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('SpaCy model loaded successfully!')"

echo.
echo Testing entropy pruning again...
python test_entropy_state.py

echo.
pause
