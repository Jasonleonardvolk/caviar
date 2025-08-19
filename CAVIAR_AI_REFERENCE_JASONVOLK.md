# AI ASSISTANT QUICK REFERENCE

## STOP - READ THIS FIRST
This is a SIMPLE git automation setup. It works perfectly. Don't suggest complications.

## THE SETUP (2 MINUTES TO UNDERSTAND)

### Core Scripts Location: `D:\Dev\kha\tools\git\`

| Script | What It Does | Command |
|--------|-------------|---------|
| **Watch-And-Push.ps1** | Auto-pushes every 2 minutes | `.\tools\git\Watch-And-Push.ps1` |
| **Watch-And-Pull.ps1** | Auto-pulls every 2 minutes | `.\tools\git\Watch-And-Pull.ps1` |
| **Create-Review-PR.ps1** | Creates PR for AI review | `.\tools\git\Create-Review-PR.ps1` |

### Quick Commands Location: `D:\Dev\kha\`

| Script | Purpose | Usage |
|--------|---------|-------|
| **p.ps1** | Manual quick push | `.\p.ps1 "commit message"` |
| **pull.ps1** | Manual pull | `.\pull.ps1` |

## FOR AI ASSISTANTS - HOW TO HELP

### When User Says "Review my code":
```powershell
# They should run:
.\tools\git\Create-Review-PR.ps1
# Creates PR #123
# Then they tell you: "Review PR #123 in Jasonleonardvolk/caviar"
```

### When User Says "Help me set up collaboration":
```powershell
# Point them to:
.\Setup-Collab-Friend.ps1  # For their collaborator
```

### Current Configuration:
- **Push Interval**: 2 minutes (was 15 seconds, changed for safety)
- **Repository**: Jasonleonardvolk/caviar
- **Main Branch**: main
- **Working Directory**: D:\Dev\kha

## WHAT NOT TO SUGGEST
❌ Pre-commit hooks (not needed)
❌ Enhanced error handling (already works)
❌ Complex parameters (keep it simple)
❌ Stash/rebase workflows (overkill)
❌ Changing the interval again (2 minutes is decided)

## WHAT TO HELP WITH
✅ Reading PRs when given PR number
✅ Reviewing code in the repo
✅ Simple PowerShell syntax fixes if needed
✅ Quick answers about git commands

## THE PHILOSOPHY
- Simple > Complex
- Working > Perfect
- Fast > Feature-rich
- 2 minutes is the sweet spot

## EXAMPLE INTERACTIONS



---
**Remember**: This setup is INTENTIONALLY SIMPLE. It works. Don't fix what isn't broken.
