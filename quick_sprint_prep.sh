#!/bin/bash
# Quick 30-minute cleanup script
# Run this to be Albert-sprint ready!

echo "🚀 30-Minute Sprint Prep"
echo "======================="

# 1. Final cleanup (10 min)
echo "Step 1: Cleaning repository..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    ./cleanup_repository.bat
else
    ./cleanup_repository.sh
fi

# 2. Tag release
echo -e "\nStep 2: Creating release tag..."
git tag -a v0.11.0-hotfix -m "Hotfix: Soliton API 500 errors resolved

- Fixed import guards and error handling  
- Added rate limiting to frontend
- Created comprehensive test suite
- Added CI/CD for concept_mesh
- Cleaned repository"

# 3. Update CI (5 min)
echo -e "\nStep 3: Optimizing CI..."
if [ -f ".github/workflows/build-concept-mesh-optimized.yml" ]; then
    mv .github/workflows/build-concept-mesh.yml .github/workflows/build-concept-mesh-full.yml
    mv .github/workflows/build-concept-mesh-optimized.yml .github/workflows/build-concept-mesh.yml
    echo "✅ CI optimized for faster PR builds"
fi

# 4. Fix README badge (2 min)
echo -e "\nStep 4: Update README badge..."
echo "⚠️  Manual step: Edit README.md and replace USERNAME/REPO with your GitHub path"

# 5. Commit everything
echo -e "\nStep 5: Committing changes..."
git add .
git commit -m "chore: Final polish - CI optimization and cleanup for v0.11.0-hotfix"

# 6. Show push commands
echo -e "\n✅ Ready to push!"
echo "Run these commands:"
echo "  git push --force origin main  # Use --force if you ran filter-repo"
echo "  git push origin v0.11.0-hotfix"
echo ""
echo "🎯 Then you're ready for the Albert sprint!"
