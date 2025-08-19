# COLLABORATION SETUP GUIDE

## For You (Jason) - Pushing Changes

### Already Set Up:
```powershell
# Auto-push every 15 seconds
.\tools\git\Watch-And-Push.ps1

# Manual quick push
.\p.ps1 "message"
```

---

## For Your Friend (Reviewer) - Receiving Changes

### One-Time Setup:
Send them this script: `Setup-Collab-Friend.ps1`

They run:
```powershell
# Clone and set up (they run ONCE)
.\Setup-Collab-Friend.ps1 -CloneDir "D:\Dev\caviar" -StartWatching
```

### Daily Use:
```powershell
# Auto-pull every 15 seconds (matches your push rate)
.\Watch-And-Pull.ps1

# Or manual pull
.\pull.ps1
```

---

## For AI Review (Claude/ChatGPT)

### Method 1: Create Draft PR (Recommended)
```powershell
# Creates a draft PR that AI can read via GitHub connector
.\tools\git\Create-Review-PR.ps1 -OpenInBrowser

# Then tell the AI:
# "Review PR #123 in Jasonleonardvolk/caviar"
```

### Method 2: Direct File Sharing
```powershell
# Share specific files in the conversation
Get-Content "D:\Dev\kha\tori_ui_svelte\src\routes\hologram\+page.svelte" | Set-Clipboard
# Then paste in AI chat
```

### Method 3: Create Shareable Diff
```powershell
# Generate a patch file
git diff > review.patch
# Upload review.patch to AI
```

---

## Complete Workflow Example

### You (Jason):
```powershell
# 1. Start auto-push
.\tools\git\Watch-And-Push.ps1

# 2. Work normally - everything auto-pushes every 15 seconds
```

### Your Friend:
```powershell
# 1. Start auto-pull (in their caviar folder)
.\Watch-And-Pull.ps1

# 2. They see your changes appear automatically
```

### For AI Review:
```powershell
# When you want AI to review code:
.\tools\git\Create-Review-PR.ps1

# Share the PR # with AI
# AI reviews and suggests changes
# You apply changes locally and push
```

---

## Quick Commands Reference

### Push Side (You):
- `.\Watch-And-Push.ps1` - Auto-push every 15 seconds
- `.\p.ps1` - Manual quick push
- `.\tools\git\Create-Review-PR.ps1` - Create PR for AI review

### Pull Side (Friend):
- `.\Watch-And-Pull.ps1` - Auto-pull every 15 seconds
- `.\pull.ps1` - Manual pull
- `.\status.ps1` - Check if behind remote

---

## Troubleshooting

### If auto-pull shows conflicts:
```powershell
# Friend runs:
git stash              # Save their local changes
git pull origin main   # Pull your changes
git stash pop          # Reapply their changes
```

### If PR creation fails:
```powershell
# Install GitHub CLI
winget install GitHub.cli

# Login
gh auth login
```

### To see what's happening:
```powershell
# You:
git log --oneline -n 5  # See recent pushes

# Friend:
git log --oneline origin/main -n 5  # See incoming changes
```