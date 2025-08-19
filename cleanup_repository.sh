#!/bin/bash
# Git cleanup script to fix repository hygiene issues
# Run this from the repository root

echo "🧹 Starting repository cleanup..."

# 1. Remove cached files that should be ignored
echo "📁 Removing cached files that shouldn't be in git..."

# Remove Python bytecode and cache
git rm -r --cached **/__pycache__/ 2>/dev/null || true
git rm -r --cached **/*.pyc 2>/dev/null || true
git rm -r --cached **/*.pyo 2>/dev/null || true
git rm -r --cached **/*.pyd 2>/dev/null || true

# Remove egg-info directories
git rm -r --cached **/*.egg-info/ 2>/dev/null || true
git rm -r --cached alan_core.egg-info/ 2>/dev/null || true
git rm -r --cached concept_mesh/concept_mesh.egg-info/ 2>/dev/null || true

# Remove logs
git rm -r --cached logs/ 2>/dev/null || true
git rm --cached **/*.log 2>/dev/null || true

# Remove data files that shouldn't be in git
git rm --cached data/memory_vault/vault_live.jsonl 2>/dev/null || true
git rm --cached **/*.pkl 2>/dev/null || true
git rm --cached **/*.pkl.gz 2>/dev/null || true

# Remove temporary and backup files
git rm --cached **/*.bak 2>/dev/null || true
git rm --cached **/*.backup_* 2>/dev/null || true
git rm --cached **/*.OLD_DUPLICATE 2>/dev/null || true

# Remove duplicate fix scripts
git rm -r --cached fixes/soliton_500_fixes/ 2>/dev/null || true
git rm --cached GREMLIN_*.ps1 2>/dev/null || true
git rm --cached fix_soliton_*.ps1 2>/dev/null || true
git rm --cached test_soliton_*.ps1 2>/dev/null || true

# Remove compiled Rust artifacts
git rm -r --cached concept_mesh/target/ 2>/dev/null || true

# 2. Add proper .gitignore if not already added
if [ ! -f .gitignore ]; then
    echo "Creating .gitignore..."
    cp .gitignore.template .gitignore
    git add .gitignore
fi

# 3. Commit the cleanup
echo "💾 Committing cleanup..."
git commit -m "chore: Clean up repository - remove logs, bytecode, and temporary files

- Remove Python cache and bytecode files
- Remove logs and session data
- Remove egg-info directories
- Remove duplicate fix scripts
- Add comprehensive .gitignore
- Keep only production code

This addresses repository hygiene issues identified in audit."

echo "✅ Cleanup complete!"
echo ""
echo "⚠️  IMPORTANT: For existing history with sensitive data:"
echo "   If data/memory_vault/vault_live.jsonl contained sensitive info,"
echo "   you may need to use git filter-repo to remove it from history:"
echo ""
echo "   pip install git-filter-repo"
echo "   git filter-repo --path data/memory_vault/vault_live.jsonl --invert-paths"
echo ""
echo "   Then force-push: git push --force origin main"
