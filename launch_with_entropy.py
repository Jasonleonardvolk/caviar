import os
import sys

# Set environment variables in the current Python process
os.environ['TORI_ENABLE_ENTROPY_PRUNING'] = '1'
os.environ.pop('TORI_DISABLE_ENTROPY_PRUNE', None)

# Now import and run the launcher
import enhanced_launcher

if __name__ == "__main__":
    launcher = enhanced_launcher.EnhancedUnifiedToriLauncher()
    sys.exit(launcher.launch())
