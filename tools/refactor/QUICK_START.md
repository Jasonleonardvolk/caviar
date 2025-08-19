# 🚀 Quick Refactoring Commands for 23GB Repository

For a repository this large, use these streamlined commands:

## Fastest Options (No Dry Run)

### Option 1: One-Line Command (No Backup)
```powershell
cd D:\Dev\kha
python tools\refactor\refactor_fast.py --quiet
```

### Option 2: With Backup (Recommended)
```powershell
cd D:\Dev\kha
python tools\refactor\refactor_fast.py --backup-dir "D:\Backups\KhaRefactor"
```

### Option 3: Interactive Batch File
```batch
cd D:\Dev\kha
tools\refactor\REFACTOR_NOW.bat
```

### Option 4: PowerShell with Progress
```powershell
cd D:\Dev\kha
.\tools\refactor\Refactor-Large-Repo.ps1 -BackupDir "D:\Backups\KhaRefactor"
```

## Performance Tips for 23GB Repo

1. **Use refactor_fast.py** - Optimized for large repos
2. **Close other applications** to free up RAM
3. **Run with --quiet flag** to reduce console output overhead
4. **Expect 10-30 minutes** depending on SSD/HDD speed
5. **Check the log file** afterwards in `tools\refactor\`

## What Gets Changed

- **Old**: `${IRIS_ROOT}`
- **New in .py files**: `${IRIS_ROOT}`
- **New in other files**: `${IRIS_ROOT}`

## Emergency Stop

If you need to stop and resume later:
- Press `Ctrl+C` to stop
- Run the same command again (the simple scripts don't have resume, but they're fast enough to just restart)

## After Completion

Check the log file in `tools\refactor\` for:
- List of all modified files
- Any errors encountered
- Count of replacements per file
