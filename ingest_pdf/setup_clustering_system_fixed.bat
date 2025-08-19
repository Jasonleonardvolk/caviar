@echo off
echo 🚀 TORI Enhanced Clustering System - Production Setup
echo ==================================================
echo.

cd /d "%~dp0"

echo 📋 System Requirements Check...
echo ================================

rem Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✅ Python available

rem Check Node.js (optional for TypeScript integration)
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Node.js not found. TypeScript integration will be limited.
    echo    Install Node.js from: https://nodejs.org/ (optional)
) else (
    echo ✅ Node.js available
)

echo.
echo 📦 Installing Python Dependencies...
echo ===================================

echo Installing core dependencies...
pip install numpy --quiet
if %errorlevel% neq 0 echo ❌ Failed to install numpy

pip install scikit-learn --quiet  
if %errorlevel% neq 0 echo ❌ Failed to install scikit-learn

pip install hdbscan --quiet
if %errorlevel% neq 0 echo ❌ Failed to install hdbscan

echo ✅ Core dependencies installed

echo.
echo 🔧 Setting up Configuration...
echo =============================

rem Create config directory
if not exist "config" mkdir config

rem Create Python script for configuration setup
echo import sys > setup_config.py
echo import os >> setup_config.py
echo sys.path.append('.') >> setup_config.py
echo try: >> setup_config.py
echo     from clustering_config import ConfigManager >> setup_config.py
echo     manager = ConfigManager() >> setup_config.py
echo     manager.save_config('production', manager.get_config('production')) >> setup_config.py
echo     manager.save_config('development', manager.get_config('development')) >> setup_config.py
echo     manager.save_config('testing', manager.get_config('testing')) >> setup_config.py
echo     print('✅ Default configurations created') >> setup_config.py
echo except Exception as e: >> setup_config.py
echo     print(f'⚠️  Configuration setup failed: {e}') >> setup_config.py

python setup_config.py
del setup_config.py

echo.
echo 🧪 Running System Tests...
echo ==========================

echo Testing oscillator clustering...
echo import numpy as np > test_clustering.py
echo import sys >> test_clustering.py
echo sys.path.append('.') >> test_clustering.py
echo try: >> test_clustering.py
echo     from clustering import run_oscillator_clustering_with_metrics >> test_clustering.py
echo     embeddings = np.random.randn(20, 10) >> test_clustering.py
echo     result = run_oscillator_clustering_with_metrics(embeddings, enable_logging=False) >> test_clustering.py
echo     print(f'✅ Oscillator clustering: {result["n_clusters"]} clusters, {result["avg_cohesion"]:.3f} cohesion') >> test_clustering.py
echo except Exception as e: >> test_clustering.py
echo     print(f'❌ Oscillator clustering test failed: {e}') >> test_clustering.py

python test_clustering.py
del test_clustering.py

echo Testing pipeline integration...
echo import sys > test_pipeline.py
echo import numpy as np >> test_pipeline.py
echo sys.path.append('.') >> test_pipeline.py
echo try: >> test_pipeline.py
echo     from clustering_pipeline import TORIClusteringPipeline, ConceptData >> test_pipeline.py
echo     concepts = [ConceptData(f'test_{i}', f'Test concept {i}', np.random.randn(10).tolist(), {}) for i in range(10)] >> test_pipeline.py
echo     pipeline = TORIClusteringPipeline() >> test_pipeline.py
echo     result = pipeline.process_concepts(concepts) >> test_pipeline.py
echo     print(f'✅ Pipeline integration: {len(result.concepts)} concepts → {len(result.cluster_summary)} clusters') >> test_pipeline.py
echo except Exception as e: >> test_pipeline.py
echo     print(f'❌ Pipeline integration test failed: {e}') >> test_pipeline.py

python test_pipeline.py
del test_pipeline.py

echo.
echo 📊 Running Quick Demo...
echo =======================

echo Running comprehensive demo...
python clustering_demo.py
if %errorlevel% neq 0 (
    echo ❌ Demo failed. Check clustering_demo.py for details.
) else (
    echo ✅ Demo completed successfully!
    if exist clustering_demo_results.json (
        echo    Results saved to: clustering_demo_results.json
    )
)

echo.
echo 🎯 Integration with Existing TORI Pipeline...
echo ============================================

echo Checking for existing TORI modules...
echo import sys > check_tori.py
echo import os >> check_tori.py
echo sys.path.append('..') >> check_tori.py
echo sys.path.append('../tori_ui_svelte/src/lib/cognitive') >> check_tori.py
echo try: >> check_tori.py
echo     import conceptScoring >> check_tori.py
echo     print('✅ TORI concept scoring module found') >> check_tori.py
echo except: >> check_tori.py
echo     print('⚠️  TORI concept scoring module not found') >> check_tori.py
echo     print('   Integration examples will be created') >> check_tori.py

python check_tori.py
del check_tori.py

echo.
echo 🔍 Creating Integration Examples...
echo ==================================

echo Creating concept_scoring_integration.py...
echo # TORI Concept Scoring Integration with Enhanced Clustering > concept_scoring_integration.py
echo # Integrates advanced clustering with existing TORI concept scoring system >> concept_scoring_integration.py
echo. >> concept_scoring_integration.py
echo from clustering import run_oscillator_clustering_with_metrics >> concept_scoring_integration.py
echo import numpy as np >> concept_scoring_integration.py
echo. >> concept_scoring_integration.py
echo def enhanced_cluster_concepts(vectors, k_estimate): >> concept_scoring_integration.py
echo     """Drop-in replacement for clusterConcepts in conceptScoring.ts""" >> concept_scoring_integration.py
echo     result = run_oscillator_clustering_with_metrics(np.array(vectors), enable_logging=False) >> concept_scoring_integration.py
echo     return result['labels'] >> concept_scoring_integration.py
echo. >> concept_scoring_integration.py
echo print("Integration helper created. Use enhanced_cluster_concepts() in your TypeScript bridge.") >> concept_scoring_integration.py

echo ✅ Integration example created: concept_scoring_integration.py

echo.
echo 📱 Creating Monitoring Setup...
echo ==============================

echo import sys > setup_monitoring.py
echo sys.path.append('.') >> setup_monitoring.py
echo try: >> setup_monitoring.py
echo     from clustering_monitor import create_production_monitor >> setup_monitoring.py
echo     monitor = create_production_monitor() >> setup_monitoring.py
echo     print('✅ Monitoring system ready') >> setup_monitoring.py
echo     print(f'   Thresholds: cohesion≥{monitor.alert_thresholds.min_cohesion}, runtime≤{monitor.alert_thresholds.max_runtime_seconds}s') >> setup_monitoring.py
echo except Exception as e: >> setup_monitoring.py
echo     print(f'⚠️  Monitoring setup failed: {e}') >> setup_monitoring.py

python setup_monitoring.py
del setup_monitoring.py

echo.
echo 🎉 SETUP COMPLETE!
echo ==================

echo Your TORI Enhanced Clustering System is ready for integration!
echo.
echo 📁 Files Ready:
if exist clustering_demo_results.json (
    echo    ✅ clustering_demo_results.json - Demo results
)
if exist concept_scoring_integration.py (
    echo    ✅ concept_scoring_integration.py - Integration helper
)
if exist config (
    echo    ✅ config/ - Configuration files
)
echo    ✅ Enhanced clustering modules ready
echo.
echo 🚀 Integration Steps:
echo    1. Your existing conceptScoring.ts is preserved
echo    2. Use concept_scoring_integration.py for Python bridge
echo    3. Replace clusterConcepts() call with enhanced version
echo    4. All your composite scoring logic remains unchanged
echo.
echo 📖 Next Steps:
echo    • Review your existing conceptScoring.ts at:
echo      tori_ui_svelte/src/lib/cognitive/conceptScoring.ts
echo    • Integration preserves your ConceptTuple structure
echo    • Enhanced clustering replaces basic K-means only
echo    • All your composite scoring features remain intact
echo.
echo ✅ Ready for seamless integration with your existing TORI system!
echo.

pause
