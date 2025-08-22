# Next Steps After Git Setup

## 1. Create GitHub Repository

### Option A: Manual Creation
1. Go to https://github.com/new
2. Create new repository with these settings:
   - **Repository name:** `iRis`
   - **Description:** iRis - Holographic UI with memory synthesis
   - **Visibility:** Public or Private (your choice)
   - **Important:** DO NOT initialize with README, .gitignore, or license

### Option B: Using GitHub CLI (if installed)
```bash
gh repo create iRis --public --source=. --remote=origin --push
```

## 2. Connect to GitHub

### Quick Connect (Recommended)
```powershell
# Run the connection script
.\Connect-GitHub.ps1 -CreateNew

# Or use the batch file
connect-github.bat
```

### Manual Connect
```bash
# Add remote
git remote add origin https://github.com/Jasonleonardvolk/iRis.git

# Push everything
git push -u origin main --follow-tags
```

## 3. GitHub Authentication

If you get authentication errors with HTTPS:

### Create Personal Access Token
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name like "tori-dev"
4. Select scopes: `repo` (full control)
5. Generate token and copy it
6. Use the token as your password when Git prompts

### Or Use SSH (Alternative)
```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your-email@example.com"

# Add to GitHub
# Copy the public key and add to https://github.com/settings/keys
cat ~/.ssh/id_ed25519.pub

# Change remote to SSH
git remote set-url origin git@github.com:Jasonleonardvolk/iRis.git
```

## 4. Start Development Workflow

### Create Your First Feature Branch
```powershell
# Using the workflow helper
.\Git-Workflow.ps1 branch feat renderer "holographic-modes"

# Or manually
git switch -c feat/renderer-holographic-modes
```

### Make Changes and Commit
```powershell
# Stage changes
git add src/routes/renderer/+page.svelte

# Commit with proper format
.\Git-Workflow.ps1 commit feat renderer "add holographic display modes"

# Or manually
git commit -m "feat(renderer): add holographic display modes"
```

### Push Feature Branch
```bash
git push -u origin feat/renderer-holographic-modes
```

### Create Pull Request
1. Go to https://github.com/Jasonleonardvolk/iRis
2. Click "Compare & pull request"
3. Add description and create PR

## 5. Daily Workflow

### Morning Sync
```bash
# Switch to main
git switch main

# Pull latest changes
git pull origin main

# Create new feature branch
git switch -c feat/your-feature-name
```

### During Development
```bash
# Check status frequently
git status

# Stage and commit often
git add .
git commit -m "feat(scope): description"

# Push to remote
git push
```

### End of Day
```bash
# Push all work
git push

# Switch to main
git switch main

# Check what branches you have
git branch -a
```

## 6. Release Process

### Create Release Candidate
```powershell
# When ready for testing
.\Git-Workflow.ps1 tag "iris-rc-v1.0.0-rc.1" -Message "Release candidate 1"
git push origin iris-rc-v1.0.0-rc.1
```

### Create GA Release
```powershell
# After testing passes
.\Git-Workflow.ps1 tag "iris-v1.0.0" -Message "First GA release"
git push origin iris-v1.0.0
```

## 7. Useful Commands Reference

### Check Everything
```powershell
# Full status check
.\Git-Workflow.ps1 status

# Or manually
git status
git branch -a
git tag --list
git remote -v
```

### Clean Up
```bash
# Delete merged local branches
git branch --merged | grep -v main | xargs -n 1 git branch -d

# Prune remote tracking branches
git remote prune origin
```

### Troubleshooting
```bash
# If push is rejected
git pull --rebase origin main
git push

# If you need to undo last commit
git reset --soft HEAD~1

# If you need to see what changed
git diff
git log --oneline -10
```

## 8. Team Collaboration

### Before Starting Work
```bash
# Always sync with main
git switch main
git pull origin main
```

### Code Review Process
1. Push feature branch
2. Create PR on GitHub
3. Request review
4. Address feedback
5. Merge when approved

### Keeping Feature Branch Updated
```bash
# While on feature branch
git merge main
# or
git rebase main
```

## Important Reminders

- **Never commit directly to main** - always use feature branches
- **Run tests before pushing** - hooks will catch issues
- **Use descriptive commit messages** - follow the convention
- **Tag important milestones** - makes rollback easier
- **Pull before starting work** - avoid conflicts
- **Commit often** - smaller commits are easier to review

## Quick Help

```powershell
# Show all workflow options
.\Git-Workflow.ps1 help

# Run end-to-end tests
.\tools\release\Verify-EndToEnd.ps1

# Check conventions
code GIT_CONVENTIONS.md
```

## Support

- Git conventions: See `GIT_CONVENTIONS.md`
- Workflow issues: Run `.\Git-Workflow.ps1 help`
- Build issues: Check `README.md`
- GitHub docs: https://docs.github.com
