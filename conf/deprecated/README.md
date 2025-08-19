# Deprecated Configuration Files

This directory contains configuration files that are no longer canonical but are preserved for reference.

## Deprecated Files

### Soliton Memory Configurations
- **soliton_memory_config_aligned.yaml** - Identical copy of canonical config, deprecated to avoid confusion
- **soliton_memory_config_consolidated.yaml** - Simplified/incomplete version, superseded by canonical config

### Lattice Configurations  
- **lattice_config_updated.yaml** - Older version missing advanced features like detailed dark soliton support
- **old_lattice_config.yaml** - Legacy FDTD simulation config, replaced by modern soliton memory system

## Migration Notes

**Do not use these files in production.** They are kept for:
- Historical reference
- Understanding evolution of configuration schema
- Backup in case specific settings need to be recovered

## Canonical Configurations

Use these files instead:
- `../soliton_memory_config.yaml` - Complete soliton memory system configuration
- `../lattice_config.yaml` - Complete lattice and topology configuration

## Removal Schedule

These files may be removed in future versions. If you need any specific settings from these deprecated configs, migrate them to the canonical configurations.
