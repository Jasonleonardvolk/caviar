@echo off
echo 🚀 TORI Enhanced Clustering System Demo
echo =======================================
echo.

cd /d "%~dp0"

echo 📋 Checking Python dependencies...
python -c "import numpy; print('✅ NumPy available')" 2>nul || (echo "❌ NumPy not found. Installing..." && pip install numpy)
python -c "import sklearn; print('✅ Scikit-learn available')" 2>nul || (echo "❌ Scikit-learn not found. Installing..." && pip install scikit-learn)
python -c "import hdbscan; print('✅ HDBSCAN available')" 2>nul || (echo "❌ HDBSCAN not found. Installing..." && pip install hdbscan)

echo.
echo 🔬 Running comprehensive clustering demo...
echo.

python clustering_demo.py

echo.
echo 📊 Demo completed! Check the following files:
echo   - clustering_demo_results.json (detailed results)
echo   - clustering_integration_example.ts (TypeScript usage)
echo   - README_CLUSTERING.md (complete documentation)
echo.

echo 🎯 Next steps:
echo   1. Review the benchmark results above
echo   2. Run 'run_clustering_integration_test.bat' to test TypeScript integration
echo   3. Integrate with your existing TORI pipeline using the examples
echo.

pause
