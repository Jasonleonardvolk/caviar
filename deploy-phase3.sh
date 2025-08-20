#!/bin/bash
# Phase 3 Deployment Script for macOS/Linux

cd D:/Dev/kha/tori_ui_svelte

echo "===================================================="
echo "      iRis PHASE 3 - VERCEL DEPLOYMENT"
echo "===================================================="
echo ""

# Step 1: Install Vercel adapter
echo "[1] Installing Vercel adapter..."
npm install -D @sveltejs/adapter-vercel
echo "    ✓ Done"

# Step 2: Test build
echo ""
echo "[2] Testing build..."
if npm run build; then
    echo "    ✓ Build successful"
else
    echo "    ✗ Build failed"
    exit 1
fi

# Step 3: Git operations
echo ""
echo "[3] Git status:"
git status --short

echo ""
read -p "Commit and push to GitHub? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    git add .
    git commit -m "iRis launch: recorder + pricing + stripe checkout"
    git push origin main
    echo "    ✓ Pushed to GitHub"
fi

echo ""
echo "===================================================="
echo "           VERCEL DEPLOYMENT STEPS"
echo "===================================================="
echo ""
echo "1. Go to: https://vercel.com"
echo "2. Add New Project → Import caviar repo"
echo "3. Set root directory: tori_ui_svelte"
echo "4. Add environment variables:"
echo "   - STRIPE_SECRET_KEY"
echo "   - STRIPE_PRICE_PLUS"
echo "   - STRIPE_PRICE_PRO"
echo "   - STRIPE_SUCCESS_URL"
echo "   - STRIPE_CANCEL_URL"
echo "5. Deploy!"
echo ""
echo "===================================================="