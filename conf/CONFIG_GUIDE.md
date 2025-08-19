# TORI Configuration Guide

This document defines the canonical configuration files for the TORI Soliton Memory System and explains their purpose and usage.

## Canonical Configuration Files

### Production Configurations

#### 1. `soliton_memory_config.yaml` - **CANONICAL SOLITON MEMORY CONFIG**
**Purpose**: Complete configuration for the soliton memory system
**Status**: Production canonical - all code should reference this file
**Features**:
- System-wide settings (lattice size, limits, version info)
- Soliton memory parameters (phase mechanics, decay rates, heat management)
- Dark soliton configuration (collision resolution, suppression modes)
- Memory crystallization (fusion, fission, migration parameters)
- Topology morphing and switching policies
- Nightly growth engine task configuration
- Comfort analysis and automated responses
- Safety and blowup detection
- Comprehensive monitoring and logging
- Performance tuning parameters
- Integration settings (MCP bridge, APIs, WebSocket)

#### 2. `lattice_config.yaml` - **CANONICAL LATTICE CONFIG**
**Purpose**: Complete lattice topology and dynamics configuration
**Status**: Production canonical - all lattice operations should use this
**Features**:
- Core lattice topology definitions (kagome, hexagonal, square, small_world, all_to_all)
- Enhanced dark soliton support with collision resolution
- Nightly consolidation phase configuration
- Topology morphing and blending parameters
- Comfort optimization with automated responses
- Soliton voting system for memory persistence
- Memory lifecycle management
- Dynamic topology policy based on system state
- Continuous optimization settings
- Performance and safety parameters
- Comprehensive event logging

### Example/Template Configurations

#### 3. `beyond_config_templates.yaml`
**Purpose**: Templates and examples for advanced configurations
**Status**: Reference/example - not loaded by default
**Usage**: Copy sections to canonical configs as needed

### Deprecated Configurations

All deprecated configuration files have been moved to `./deprecated/` directory. 
**Do not use deprecated files in production code.**

## Configuration Loading Priority

The system should load configurations in this order:

1. **Primary**: `soliton_memory_config.yaml` for all soliton memory operations
2. **Primary**: `lattice_config.yaml` for all lattice and topology operations  
3. **Optional**: Environment-specific overrides (if implemented)
4. **Fallback**: Hard-coded defaults in source code

## Development Guidelines

### Adding New Configuration Options

1. **For soliton memory features**: Add to `soliton_memory_config.yaml`
2. **For lattice/topology features**: Add to `lattice_config.yaml`
3. **Maintain backward compatibility** when modifying existing options
4. **Document new options** with comments in the YAML files
5. **Update this guide** when adding major new sections

### Configuration Validation

All configuration loading code should:
- Validate required fields are present
- Provide sensible defaults for optional fields
- Log warnings for deprecated options
- Fail gracefully with clear error messages

### Testing

- Test with canonical configurations before deployment
- Verify deprecated configs are not accidentally loaded
- Test configuration validation and error handling
- Ensure defaults work when config files are missing

## Code References

### Where Configurations Are Used

**Soliton Memory Config** (`soliton_memory_config.yaml`):
- `python/core/soliton_memory.py` - Main soliton memory system
- `python/core/fractal_soliton_memory.py` - Enhanced memory engine
- `api/routes/soliton_production.py` - API endpoints
- Various memory management modules

**Lattice Config** (`lattice_config.yaml`):
- `python/core/hot_swap_laplacian.py` - Topology switching
- `python/core/chaos_control_layer.py` - Chaos control and optimization
- `python/core/oscillator_lattice.py` - Basic lattice operations
- Various topology and dynamics modules

### Configuration Loading Pattern

```python
# Recommended pattern for loading canonical configs
import yaml
from pathlib import Path

def load_soliton_config():
    config_path = Path(__file__).parent / "conf" / "soliton_memory_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_lattice_config():
    config_path = Path(__file__).parent / "conf" / "lattice_config.yaml" 
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

## Migration from Deprecated Configs

If you were using any of the deprecated configuration files:

### From `soliton_memory_config_aligned.yaml` or `soliton_memory_config_consolidated.yaml`
- **Action**: Switch to `soliton_memory_config.yaml`
- **Compatibility**: `soliton_memory_config.yaml` includes all features from other versions
- **No changes needed** - canonical config is feature-complete

### From `lattice_config_updated.yaml` 
- **Action**: Switch to `lattice_config.yaml`
- **Benefits**: Enhanced dark soliton support, better nightly consolidation, comfort optimization
- **Migration**: Review any custom settings and merge into canonical config

### From `old_lattice_config.yaml`
- **Action**: Complete migration required to `lattice_config.yaml`
- **Reason**: Completely different schema (legacy FDTD vs modern soliton memory system)
- **Process**: Identify equivalent modern settings for any custom parameters

## Configuration Optimization

The canonical configurations have been optimized for:

### Performance
- Efficient batch sizes and update intervals
- Appropriate memory limits and cleanup schedules
- GPU acceleration options where available
- Sparse matrix thresholds for large systems

### Stability  
- Conservative safety thresholds to prevent blowup
- Fallback topologies for emergency situations
- Gradual topology transitions to avoid instability
- Comprehensive error handling and recovery

### Functionality
- Complete feature coverage for all system capabilities
- Sensible defaults that work out-of-the-box
- Flexible policy configurations for different use cases
- Extensive monitoring and diagnostics

## Future Plans

### Configuration System Enhancements
- Environment variable overrides
- Runtime configuration hot-reloading
- Configuration validation schemas
- Web-based configuration management interface

### Deprecation Timeline
- **v2.1**: Deprecated configs moved to deprecated/ (completed)
- **v2.2**: Warning messages when deprecated configs detected
- **v2.3**: Remove deprecated config loading support  
- **v3.0**: Remove deprecated config files entirely

## Support

For configuration questions or issues:
1. Check this guide first
2. Review comments in the canonical YAML files
3. Check existing code for usage examples
4. Consult system logs for configuration loading messages

---

**Remember**: Always use the canonical configurations for production systems. The deprecated configurations may have missing features, incorrect defaults, or compatibility issues.
