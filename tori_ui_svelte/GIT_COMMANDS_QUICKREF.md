# 🚀 iRis Git Workflow - Quick Command Reference

## 📍 **Navigate to Project Directory**

### From Windows Command Prompt (CMD):
```cmd
cd /d D:\Dev\kha\tori_ui_svelte
```

### From PowerShell:
```powershell
Set-Location D:\Dev\kha\tori_ui_svelte
# or shorter:
cd D:\Dev\kha\tori_ui_svelte
```

### From Windows Run Dialog (Win+R):
```
cmd /k "cd /d D:\Dev\kha\tori_ui_svelte"
```

---

## 🎯 **Essential Commands**

### **1️⃣ One-Time Setup** (Run this first!)
```cmd
cd /d D:\Dev\kha\tori_ui_svelte
.\setup-git-hooks.bat
```

### **2️⃣ Test the System**
```cmd
cd /d D:\Dev\kha\tori_ui_svelte
.\test-git-workflow.bat
```

### **3️⃣ Check Project Status**
```cmd
cd /d D:\Dev\kha\tori_ui_svelte
.\git-status.bat
```

---

## 🎮 **Interactive Menu** (Easiest!)

### Run the All-in-One Menu:
```cmd
D:\Dev\kha\tori_ui_svelte\GIT-WORKFLOW-MENU.bat
```
*This opens an interactive menu with all options!*

---

## 🖱️ **Create Desktop Shortcuts**

### Run this once to create desktop shortcuts:
```powershell
cd D:\Dev\kha\tori_ui_svelte
powershell -ExecutionPolicy Bypass -File "Create-GitWorkflowShortcuts.ps1"
```

This creates 3 shortcuts on your desktop:
- **iRis Git Workflow** - Opens the interactive menu
- **iRis Quick Ship** - Build and deploy
- **iRis Project Folder** - Opens the project directory

---

## 💻 **Direct PowerShell Commands**

### Without changing directory:
```powershell
# Setup hooks
& "D:\Dev\kha\tori_ui_svelte\setup-git-hooks.bat"

# Test system
& "D:\Dev\kha\tori_ui_svelte\test-git-workflow.bat"

# Check status
& "D:\Dev\kha\tori_ui_svelte\git-status.bat"

# Use Git workflow directly
& "D:\Dev\kha\tori_ui_svelte\tools\git\Git-Workflow.ps1" status
```

---

## 📋 **Common Git Workflow Operations**

### After navigating to project directory:

#### Create Feature Branch:
```powershell
.\tools\git\Git-Workflow.ps1 branch feat renderer "cool-feature" -Push
```

#### Tag Internal Build:
```powershell
.\tools\git\Git-Workflow.ps1 internal "0.1.2" -Message "Demo build" -Push
```

#### Create Release Candidate:
```powershell
.\tools\git\Git-Workflow.ps1 rc "1.0.0-rc.1" -Message "Testing release" -Push
```

#### Ship Production Release:
```powershell
.\tools\git\Git-Workflow.ps1 release "1.0.0" -Message "iRis 1.0 GA" -Push
```

---

## 🔧 **Troubleshooting**

### If scripts won't run:
```powershell
# Allow script execution (run as admin)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### If "not recognized" error:
```cmd
# Make sure you're in the right directory
cd /d D:\Dev\kha\tori_ui_svelte
dir *.bat
```

### If Git hooks aren't working:
```cmd
cd /d D:\Dev\kha\tori_ui_svelte
git config core.hooksPath .githooks
```

---

## 📌 **Pro Tips**

1. **Pin to taskbar**: Drag `GIT-WORKFLOW-MENU.bat` to your taskbar for quick access

2. **Add to PATH**: Add `D:\Dev\kha\tori_ui_svelte` to your system PATH to run commands from anywhere

3. **Create alias in PowerShell profile**:
   ```powershell
   # Add to $PROFILE
   function iris { Set-Location D:\Dev\kha\tori_ui_svelte }
   function iris-menu { & "D:\Dev\kha\tori_ui_svelte\GIT-WORKFLOW-MENU.bat" }
   ```

4. **VS Code terminal**: Open integrated terminal in VS Code and commands work directly

---

## 📁 **File Locations**

| What | Where |
|------|-------|
| Project Root | `D:\Dev\kha\tori_ui_svelte\` |
| Git Hooks | `D:\Dev\kha\tori_ui_svelte\.githooks\` |
| Workflow Scripts | `D:\Dev\kha\tori_ui_svelte\tools\git\` |
| Release Scripts | `D:\Dev\kha\tori_ui_svelte\tools\release\` |
| Quick Launchers | `D:\Dev\kha\tori_ui_svelte\*.bat` |

---

## ⚡ **Fastest Way to Start**

1. **Press** `Win+R`
2. **Type**: `cmd`
3. **Paste**: `D:\Dev\kha\tori_ui_svelte\GIT-WORKFLOW-MENU.bat`
4. **Press** Enter

You're now in the interactive menu! 🎉