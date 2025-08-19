# PowerShell script to create o4.zip with all physics implementation files

# Change to the kha directory
cd "C:\Users\jason\Desktop\tori\kha"

# Create the zip file with all O4 physics files
Compress-Archive -Path @(
    # Core physics implementations
    "python\core\strang_integrator.py",
    "python\core\physics_hot_swap_laplacian.py",
    "python\core\physics_oscillator_lattice.py",
    "python\core\physics_blowup_harness.py",
    "python\core\physics_instrumentation.py",
    "python\core\bdg_solver.py",
    
    # Test files
    "tests\test_physics_validation.py",
    
    # Patches
    "patches\25_comfort_feedback_integration.py",
    
    # All O3 files (from previous work)
    "patches\01_fix_soliton_memory_complete.rs",
    "patches\02_fix_lattice_topology_complete.rs",
    "patches\03_fix_memory_fusion_fission_complete.py",
    "patches\04_fix_hot_swap_energy_harvesting.py",
    "patches\05_fix_topology_policy_config_driven.py",
    "patches\06_fix_nightly_growth_scheduler.py",
    "patches\08_fix_comfort_analysis_feedback.rs",
    "patches\09_complete_memory_crystallization.py",
    "patches\10_fix_blowup_harness_safety.py",
    "patches\11_logging_configuration.py",
    "patches\15_enhanced_benchmarking_complete.py",
    "patches\20_dark_soliton_mode.rs",
    "patches\21_enhanced_lattice_topologies.rs",
    "patches\22_soliton_interactions.py",
    "patches\23_nightly_consolidation_enhanced.py",
    "patches\24_dark_soliton_python.py",
    "tests\test_end_to_end_morphing.py",
    "tests\test_full_nightly_cycle.py",
    "tests\test_get_global_lattice_import.py",
    "conf\soliton_memory_config_aligned.yaml",
    "conf\enhanced_lattice_config.yaml",
    
    # Documentation
    "O2_COMPREHENSIVE_FIX_PLAN.md",
    "O2_COMPLETE_FIX_SUMMARY.md",
    "ADVANCED_SOLITON_FEATURES_COMPLETE.md",
    "O3_TOUCHUP_FIXES_COMPLETE.md",
    "O4_PHYSICS_IMPLEMENTATION_COMPLETE.md"
) -DestinationPath "C:\Users\jason\Desktop\o4.zip" -Force

Write-Host "Successfully created o4.zip with all physics implementation files!" -ForegroundColor Green

# One-liner version:
# cd "C:\Users\jason\Desktop\tori\kha"; Compress-Archive -Path @("python\core\strang_integrator.py","python\core\physics_hot_swap_laplacian.py","python\core\physics_oscillator_lattice.py","python\core\physics_blowup_harness.py","python\core\physics_instrumentation.py","python\core\bdg_solver.py","tests\test_physics_validation.py","patches\25_comfort_feedback_integration.py","patches\01_fix_soliton_memory_complete.rs","patches\02_fix_lattice_topology_complete.rs","patches\03_fix_memory_fusion_fission_complete.py","patches\04_fix_hot_swap_energy_harvesting.py","patches\05_fix_topology_policy_config_driven.py","patches\06_fix_nightly_growth_scheduler.py","patches\08_fix_comfort_analysis_feedback.rs","patches\09_complete_memory_crystallization.py","patches\10_fix_blowup_harness_safety.py","patches\11_logging_configuration.py","patches\15_enhanced_benchmarking_complete.py","patches\20_dark_soliton_mode.rs","patches\21_enhanced_lattice_topologies.rs","patches\22_soliton_interactions.py","patches\23_nightly_consolidation_enhanced.py","patches\24_dark_soliton_python.py","tests\test_end_to_end_morphing.py","tests\test_full_nightly_cycle.py","tests\test_get_global_lattice_import.py","conf\soliton_memory_config_aligned.yaml","conf\enhanced_lattice_config.yaml","O2_COMPREHENSIVE_FIX_PLAN.md","O2_COMPLETE_FIX_SUMMARY.md","ADVANCED_SOLITON_FEATURES_COMPLETE.md","O3_TOUCHUP_FIXES_COMPLETE.md","O4_PHYSICS_IMPLEMENTATION_COMPLETE.md") -DestinationPath "C:\Users\jason\Desktop\o4.zip" -Force
