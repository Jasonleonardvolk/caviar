@echo off
echo.
echo Continuing installation...
echo.

REM Continue with transformers and sentence-transformers
echo Installing transformers 4.44.2...
pip install transformers==4.44.2

echo.
echo Installing sentence-transformers 3.0.1...
pip install sentence-transformers==3.0.1

echo.
echo Installing spacy 3.7.5...
pip install spacy==3.7.5

echo.
echo ========================================
echo Testing imports...
echo ========================================
python -c "import numpy; print(f'numpy {numpy.__version__} - OK')"
python -c "import scipy; print(f'scipy {scipy.__version__} - OK')"
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__} - OK')"
python -c "import transformers; print(f'transformers {transformers.__version__} - OK')"
python -c "import sentence_transformers; print(f'sentence-transformers {sentence_transformers.__version__} - OK')"
python -c "import spacy; print(f'spacy {spacy.__version__} - OK')"

echo.
echo Testing entropy pruning...
python -c "import sys; sys.path.insert(0, '.'); from ingest_pdf.entropy_prune import entropy_prune; print('entropy_prune imports successfully!')"

echo.
echo ========================================
echo Complete test:
echo ========================================
python test_entropy_state.py

pause
