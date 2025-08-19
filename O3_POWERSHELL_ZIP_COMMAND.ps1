# PowerShell script to create o3.zip with all O2 Soliton Memory implementation files
# Total: 30 files from the git staging list

# Change to the kha directory
cd "C:\Users\jason\Desktop\tori\kha"

# Create the zip file with all the programming files
Compress-Archive -Path @(
    # Documentation files (7 total)
    "ADVANCED_SOLITON_FEATURES_COMPLETE.md",
    "O2_CODE_REVIEW_FIXES_SUMMARY.md",
    "O2_COMPLETE_FIX_SUMMARY.md", 
    "O2_COMPREHENSIVE_FIX_PLAN.md",
    
    # Configuration files (3 total)
    "conf\enhanced_lattice_config