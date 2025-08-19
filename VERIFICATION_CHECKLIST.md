# TORI Self-Transformation Verification Checklist

## Quick Verification Steps

Run these commands to verify the implementation:

### 1. Check all files exist
```bash
dir /s /b safety meta_genome meta goals audit *.py *.json *.md
```

### 2. Run the startup check
```bash
python startup_self_transformation.py
```

Expected output:
```
✓ Constitutional Safety
✓ Critic Consensus
✓ Energy Budget
✓ Analogical Transfer
✓ Audit Logger
Components: 5/5 operational
Status: READY FOR SELF-TRANSFORMATION
```

### 3. Run the demo
```bash
python demo_self_transformation.py
```

### 4. Run the test suite
```bash
python test_self_transformation.py
```

### 5. Check audit log creation
```bash
type audit\events.log
```

## File Checklist

- [ ] `/safety/constitution.py` - Constitutional enforcer
- [ ] `/safety/constitution.json` - Default configuration
- [ ] `/safety/toriconstitution.schema.json` - JSON schema
- [ ] `/meta_genome/critics/aggregation.py` - Critic consensus
- [ ] `/meta/sandbox/runner.py` - Sandbox runner
- [ ] `/meta/energy_budget.py` - Energy management
- [ ] `/goals/analogical_transfer.py` - Knowledge transfer
- [ ] `/audit/logger.py` - Audit logging
- [ ] `/config/resource_limits.toml` - Resource configuration
- [ ] All `__init__.py` files for proper imports

## Integration Readiness

- [ ] Constitution limits appropriate for hardware
- [ ] Critic parameters initialized
- [ ] Energy thresholds configured
- [ ] Audit directory writable
- [ ] Git available for sandbox operations

## Next Steps

1. Run `START_SELF_TRANSFORMATION.bat` to initialize
2. Monitor `self_transformation_state.json` for system state
3. Check `audit/events.log` for operation history
4. Integrate with existing TORI cognitive systems

## Troubleshooting

If components fail to initialize:

1. **Import Errors**: Ensure you're in the correct directory
2. **Constitution Error**: Check `constitution.json` is valid JSON
3. **Permission Error**: Ensure write access to audit directory
4. **Git Error**: Install Git and ensure it's in PATH
5. **Resource Error**: psutil module may need installation

## Success Indicators

- All 5 components show ✓ on startup
- Demo runs without errors
- Tests pass (7/7 or more)
- Audit log contains startup event
- Energy budget shows positive efficiency

---

Once all checks pass, TORI's self-transformation system is ready for integration with the broader cognitive architecture! 🚀
