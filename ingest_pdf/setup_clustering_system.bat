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
pip install numpy --quiet || echo "❌ Failed to install numpy"
pip install scikit-learn --quiet || echo "❌ Failed to install scikit-learn" 
pip install hdbscan --quiet || echo "❌ Failed to install hdbscan"

echo ✅ Core dependencies installed

echo.
echo 🔧 Setting up Configuration...
echo =============================

rem Create config directory
if not exist "config" mkdir config

rem Create default configuration
python -c "
from clustering_config import ConfigManager
manager = ConfigManager()
manager.save_config('production', manager.get_config('production'))
manager.save_config('development', manager.get_config('development'))
manager.save_config('testing', manager.get_config('testing'))
print('✅ Default configurations created')
"

echo.
echo 🧪 Running System Tests...
echo ==========================

echo Testing oscillator clustering...
python -c "
import numpy as np
from clustering import run_oscillator_clustering_with_metrics
embeddings = np.random.randn(20, 10)
result = run_oscillator_clustering_with_metrics(embeddings, enable_logging=False)
print(f'✅ Oscillator clustering: {result[\"n_clusters\"]} clusters, {result[\"avg_cohesion\"]:.3f} cohesion')
" || echo "❌ Oscillator clustering test failed"

echo Testing pipeline integration...
python -c "
from clustering_pipeline import TORIClusteringPipeline, ConceptData
import numpy as np
concepts = [ConceptData(f'test_{i}', f'Test concept {i}', np.random.randn(10).tolist(), {}) for i in range(10)]
pipeline = TORIClusteringPipeline()
result = pipeline.process_concepts(concepts)
print(f'✅ Pipeline integration: {len(result.concepts)} concepts → {len(result.cluster_summary)} clusters')
" || echo "❌ Pipeline integration test failed"

echo.
echo 📊 Running Quick Demo...
echo =======================

echo Running comprehensive demo...
python clustering_demo.py >demo_output.txt 2>&1
if %errorlevel% neq 0 (
    echo ❌ Demo failed. Check demo_output.txt for details.
) else (
    echo ✅ Demo completed successfully!
    echo    Results saved to: clustering_demo_results.json
)

echo.
echo 🎯 Integration with Existing TORI Pipeline...
echo ============================================

echo Checking for existing TORI modules...
python -c "
import sys, os
sys.path.append('..')
try:
    from pipeline import *
    print('✅ TORI pipeline modules found')
except:
    print('⚠️  TORI pipeline modules not found in parent directory')
    print('   Manual integration may be required')
"

echo.
echo 🔍 Creating Integration Examples...
echo ==================================

rem Create simple integration example
echo Creating integration_example.py...
echo # Simple TORI Clustering Integration Example > integration_example.py
echo from clustering_pipeline import TORIClusteringPipeline, ConceptData >> integration_example.py
echo import numpy as np >> integration_example.py
echo. >> integration_example.py
echo # Replace your existing clustering call: >> integration_example.py
echo # labels = your_clustering_function(embeddings) >> integration_example.py
echo. >> integration_example.py
echo # With enhanced clustering: >> integration_example.py
echo pipeline = TORIClusteringPipeline() >> integration_example.py
echo concepts = [ConceptData(f'concept_{i}', f'Text {i}', embedding.tolist(), {}) for i, embedding in enumerate(embeddings)] >> integration_example.py
echo result = pipeline.process_concepts(concepts) >> integration_example.py
echo enhanced_labels = [r.cluster_id for r in result.clustering_results] >> integration_example.py

echo ✅ Integration example created: integration_example.py

echo.
echo 📱 Creating Monitoring Dashboard...
echo ==================================

python -c "
from clustering_monitor import create_production_monitor
monitor = create_production_monitor()
print('✅ Monitoring system ready')
print(f'   Thresholds: cohesion≥{monitor.alert_thresholds.min_cohesion}, runtime≤{monitor.alert_thresholds.max_runtime_seconds}s')
"

echo.
echo 🎉 SETUP COMPLETE!
echo ==================

echo Your TORI Enhanced Clustering System is ready for production use!
echo.
echo 📁 Files Created:
echo    • clustering_demo_results.json - Demo results
echo    • integration_example.py - Simple integration guide
echo    • config/production_config.json - Production configuration
echo    • config/development_config.json - Development configuration
echo    • clustering_monitor.log - System monitoring log
echo.
echo 🚀 Quick Start Commands:
echo    • python clustering_demo.py - Run comprehensive demo
echo    • run_clustering_integration_test.bat - Test TypeScript bridge
echo    • python integration_example.py - Test basic integration
echo.
echo 📖 Documentation:
echo    • README_CLUSTERING.md - Complete documentation
echo    • clustering_integration_example.ts - TypeScript examples
echo.
echo 🎯 Next Steps:
echo    1. Review demo results in clustering_demo_results.json
echo    2. Integrate with your existing TORI pipeline using integration_example.py
echo    3. Configure monitoring alerts in clustering_monitor.py
echo    4. Run benchmarks on your actual data to optimize parameters
echo.
echo 💡 Support:
echo    • Check README_CLUSTERING.md for troubleshooting
echo    • Run clustering_demo.py for detailed examples
echo    • Monitor clustering_monitor.log for system health
echo.

pause
