#!/bin/bash
# Install Stripe dependency
echo "Installing Stripe SDK..."
cd D:/Dev/kha/tori_ui_svelte
npm install stripe

echo "iRis Setup Complete!"
echo "Next steps:"
echo "1. Set up your Stripe account at https://dashboard.stripe.com"
echo "2. Create test products and prices"
echo "3. Update .env with your actual Stripe keys"
echo "4. Run: npm run dev"
echo "5. Visit: http://localhost:5173/hologram-studio"