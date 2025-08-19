# README - BridgeCommunication.exe Diagnostic and Remediation Tools
# ================================================================

## Quick Start Guide

These PowerShell tools help identify and neutralize the BridgeCommunication.exe spawning issue.
Run all scripts as Administrator for full visibility and functionality.

## Tools Overview

### 1. Inspect-BridgeCommunication.ps1
**Purpose:** Fast fingerprint to identify who's spawning BridgeCommunication.exe
**Location:** D:\Dev\kha\tools\diagnostics\

**Usage:**
```powershell
.\Inspect-BridgeCommunication.ps1
```

**What it shows:**
- Full path to each BridgeCommunication.exe
- Company/Product from file metadata
- Digital signature details
- Parent process name and command line
- Memory usage per process
- CSV snapshot saved with timestamp

### 2. Hunt-BridgePersistence.ps1
**Purpose:** Find how BridgeCommunication.exe persists through reboots
**Location:** D:\Dev\kha\tools\diagnostics\

**Usage:**
```powershell
.\Hunt-BridgePersistence.ps1
# Or for a different process:
.\Hunt-BridgePersistence.ps1 -ExeName "SomeOther.exe"
```

**What it checks:**
- File locations in Program Files, ProgramData, AppData
- Windows Services referencing the exe
- Scheduled Tasks that launch it
- Registry Run keys (HKLM/HKCU)
- Startup folder items

### 3. Quarantine-BridgeCommunication.ps1
**Purpose:** Kill processes and prevent them from running again
**Location:** D:\Dev\kha\tools\remediation\

**Usage:**
```powershell
# Preview what would happen (recommended first)
.\Quarantine-BridgeCommunication.ps1 -DryRun

# Execute quarantine
.\Quarantine-BridgeCommunication.ps1

# Skip firewall rules
.\Quarantine-BridgeCommunication.ps1 -SkipFirewall

# Verbose output
.\Quarantine-BridgeCommunication.ps1 -Verbose
```

**What it does:**
- Terminates all running processes
- Disables related Windows services
- Blocks file execution via NTFS ACLs
- Renames files to .quarantine
- Creates outbound firewall blocks
- Disables scheduled tasks

### 4. Kill-ProcessSwarm.ps1
**Purpose:** Generic tool to kill any process swarm
**Location:** D:\Dev\kha\tools\diagnostics\

**Usage:**
```powershell
# Interactive mode (asks for confirmation)
.\Kill-ProcessSwarm.ps1 -ProcessName "BridgeCommunication"

# Force mode (no confirmation)
.\Kill-ProcessSwarm.ps1 -ProcessName "BridgeCommunication" -Force

# Wait for termination with verbose output
.\Kill-ProcessSwarm.ps1 -ProcessName "BridgeCommunication" -WaitForTermination -Verbose
```

## Recommended Workflow (5 minutes)

1. **Identify the culprit:**
   ```powershell
   cd D:\Dev\kha\tools\diagnostics
   .\Inspect-BridgeCommunication.ps1
   ```
   Look for Company, Product, and ParentName in the output.

2. **Find persistence mechanisms:**
   ```powershell
   .\Hunt-BridgePersistence.ps1
   ```
   Note which services, tasks, or registry keys are involved.

3. **Preview quarantine actions:**
   ```powershell
   cd ..\remediation
   .\Quarantine-BridgeCommunication.ps1 -DryRun
   ```

4. **Execute quarantine if safe:**
   ```powershell
   .\Quarantine-BridgeCommunication.ps1
   ```

5. **Verify no respawns:**
   ```powershell
   cd ..\diagnostics
   .\Inspect-BridgeCommunication.ps1
   ```

6. **Reboot and re-verify**

## Reverting Quarantine Actions

If you need to restore BridgeCommunication.exe:

1. **Re-enable services:**
   ```powershell
   sc.exe config [ServiceName] start= auto
   sc.exe start [ServiceName]
   ```

2. **Remove ACL blocks:**
   ```powershell
   icacls "C:\Path\To\BridgeCommunication.exe" /remove:d *S-1-1-0
   ```

3. **Rename files back:**
   ```powershell
   Rename-Item "BridgeCommunication.exe.quarantine" "BridgeCommunication.exe"
   ```

4. **Remove firewall rules:**
   - Open Windows Defender Firewall with Advanced Security
   - Delete rules starting with "Block-BridgeCommunication-"

5. **Re-enable scheduled tasks:**
   ```powershell
   Enable-ScheduledTask -TaskName "TaskNameHere"
   ```

## Likely Origins

Based on the filename and behavior pattern (multiple ~3MB instances), this could be:
- Device bridge/helper for peripherals (3D mice, spatial controllers)
- Display management software (especially for specialized monitors)
- Third-party control panel utilities
- SDK/runtime for specialized hardware

The parent process identified by Inspect-BridgeCommunication.ps1 will reveal the vendor.

## Additional Investigation

For deeper forensics, enable process creation auditing:
```powershell
auditpol /set /subcategory:"Process Creation" /success:enable /failure:enable
```

Then check Event Viewer:
- Windows Logs > Security
- Filter for Event ID 4688
- Look for BridgeCommunication.exe entries

## Safety Notes

- Always run diagnostics first before remediation
- Use -DryRun flag to preview changes
- These tools are reversible - quarantine doesn't delete files
- Keep the CSV snapshots for incident documentation

## Support

For issues or questions about these tools, reference:
- Initial snapshot: BridgeComm_snapshot_[timestamp].csv
- Parent process information from Inspect-BridgeCommunication.ps1
- Service/task names from Hunt-BridgePersistence.ps1

Last Updated: 2025