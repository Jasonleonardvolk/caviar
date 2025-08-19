@echo off
echo ======================================
echo Installing Entropy Pruning Dependencies
echo ======================================
echo.

REM Install all required packages
echo Installing sentence-transformers...
pip install sentence-transformers

echo.
echo Installing scikit-learn...
pip install scikit-learn

echo.
echo Installing numpy (if not already installed)...
pip install numpy

echo.
echo ======================================
echo Testing imports...
echo ======================================
python -c "from sentence_transformers import SentenceTransformer; print('✅ sentence_transformers OK')"
python -c "from sklearn.cluster import AgglomerativeClustering; print('✅ sklearn OK')"
python -c "import numpy; print('✅ numpy OK')"

echo.
echo ======================================
echo Verifying entropy pruning...
echo ======================================
python verify_entropy.py

echo.
echo Done! Press any key to exit...
pause
