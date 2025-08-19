# Path Refactoring Tools

## Quick Start

The simplest way to refactor all absolute paths in your repository:

```batch
cd D:\Dev\kha
tools\refactor\REFACTOR_PATHS.bat
```

This will:
1. First do a dry run showing what will be changed
2. Ask for confirmation
3. Optionally backup files
4. Perform the refactoring

## Available Scripts

### 1. mass_refactor_simple.py
The main refactoring script based on the working quick_scan approach.

**Direct usage:**
```powershell
# Dry run (preview changes without modifying files)
python tools\refactor\mass_refactor_simple.py --dry-run

# Actual refactoring with backup
python tools\refactor\mass_refactor_simple.py --backup-dir "D:\Backups\KhaRefactor"

# Resume interrupted refactoring
python tools\refactor\mass_refactor_simple.py --resume --backup-dir "D:\Backups\KhaRefactor"
```

### 2. Run-SimpleRefactor.ps1
PowerShell wrapper with convenient defaults.

```powershell
# Dry run
.\tools\refactor\Run-SimpleRefactor.ps1 -DryRun

# With backup
.\tools\refactor\Run-SimpleRefactor.ps1 -BackupDir "D:\Backups\KhaRefactor"

# Resume mode
.\tools\refactor\Run-SimpleRefactor.ps1 -Resume -BackupDir "D:\Backups\KhaRefactor"
```

### 3. REFACTOR_PATHS.bat
Interactive batch file - easiest to use!

```batch
tools\refactor\REFACTOR_PATHS.bat
```

### 4. quick_scan.py
Just scan for files containing the old path (no modifications).

```powershell
python tools\refactor\quick_scan.py
```

## What Gets Refactored

**Old Path:** `${IRIS_ROOT}`

**Replacements:**
- **Python files (.py)**: Replaced with `${IRIS_ROOT}` and adds Path import header
- **Other files (.ts, .js, .json, etc.)**: Replaced with `${IRIS_ROOT}`

**Processed Extensions:**
- Python: `.py`
- TypeScript/JavaScript: `.ts`, `.tsx`, `.js`, `.jsx`
- Web: `.svelte`, `.wgsl`
- Config: `.json`, `.yaml`, `.yml`
- Docs: `.md`, `.txt`

**Excluded Directories:**
- `.git`, `.venv`, `venv`, `node_modules`
- `dist`, `build`, `.cache`, `__pycache__`
- `.pytest_cache`, `target`, `.idea`, `.vscode`
- `tools\dawn`

## Features

- **Dry Run Mode**: Preview changes before applying
- **Backup Support**: Save original files before modification
- **Resume Capability**: Continue if interrupted
- **Progress Reporting**: Shows files as they're processed
- **Size Limit**: Skips files larger than 2MB
- **Safe Processing**: Only processes text files
- **Detailed Logging**: Creates logs in `tools\refactor\`

## Output Files

After running, check these files in `tools\refactor\`:
- `refactor_plan.csv` - List of files to be changed (dry run)
- `refactor_state.json` - Resume state
- `refactor_log_*.txt` - Detailed log of changes
- `quick_scan_results.csv` - Results from quick_scan.py

## Recommended Workflow

1. **First, scan to see what needs changing:**
   ```
   python tools\refactor\quick_scan.py
   ```

2. **Do a dry run to preview changes:**
   ```
   python tools\refactor\mass_refactor_simple.py --dry-run
   ```

3. **Run the actual refactoring with backup:**
   ```
   python tools\refactor\mass_refactor_simple.py --backup-dir "D:\Backups\KhaRefactor"
   ```

   Or just use the interactive batch file:
   ```
   tools\refactor\REFACTOR_PATHS.bat
   ```

## After Refactoring

Your code will now use relative paths:
- Python files will use `${IRIS_ROOT}` which resolves at runtime
- Other files will use `${IRIS_ROOT}` which can be resolved from environment/config

This makes your codebase portable and independent of absolute file paths!
