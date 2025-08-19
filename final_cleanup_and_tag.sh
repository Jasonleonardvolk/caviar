#!/bin/bash
# Final cleanup and tag release
# Run this after reviewing changes

echo "🧹 Starting final cleanup for v0.11.0-hotfix..."

# 1. Run git cleanup
echo "📁 Removing cached files..."
git rm -r --cached logs/ 2>/dev/null || true
git rm --cached **/*.pkl 2>/dev/null || true
git rm --cached **/*.pkl.gz 2>/dev/null || true
git rm --cached **/*.pyc 2>/dev/null || true
git rm --cached **/*.egg-info/ 2>/dev/null || true

# 2. Commit cleanup
git commit -m "chore: Final cleanup before v0.11.0-hotfix" || true

# 3. Filter history to remove bloat
echo "🔧 Filtering repository history..."
echo "This will rewrite history. Make sure you have a backup!"
read -p "Continue with git filter-repo? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Install git-filter-repo if needed
    pip install git-filter-repo
    
    # Remove large files from history
    git filter-repo --path logs/ --invert-paths
    git filter-repo --path "*.pkl" --invert-paths  
    git filter-repo --path "*.pkl.gz" --invert-paths
    git filter-repo --path data/memory_vault/ --invert-paths
    
    echo "✅ History filtered"
else
    echo "⏭️  Skipped history filtering"
fi

# 4. Tag the release
echo "🏷️  Creating tag v0.11.0-hotfix..."
git tag -a v0.11.0-hotfix -m "Hotfix: Soliton API 500 errors resolved

- Fixed import guards and error handling
- Added rate limiting to frontend
- Created comprehensive test suite
- Added CI/CD for concept_mesh
- Cleaned repository of temporary files"

echo "✅ Tag created"

# 5. Show what to push
echo ""
echo "📤 Ready to push! Run:"
echo "  git push --force origin main"
echo "  git push origin v0.11.0-hotfix"
echo ""
echo "⚠️  Note: --force is needed after filter-repo"
