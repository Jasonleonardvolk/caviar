# PHASE 2 - Local Test Checklist
# iRis Hologram Studio Testing Guide

## Prerequisites
- [ ] Phase 1 files created (plans.json, userPlan.ts, HologramRecorder.svelte, etc.)
- [ ] Stripe account created (test mode)
- [ ] Node.js and pnpm installed

## Test Execution

### 1. Install Dependencies & Run Dev
```bash
cd D:\Dev\kha\tori_ui_svelte
pnpm install
pnpm dev
```
- [ ] Server starts on http://localhost:5173
- [ ] No build errors

### 2. Test Free Plan Recording
- [ ] Navigate to `/hologram` or `/hologram-studio`
- [ ] Recorder bar shows "Free" pill
- [ ] Button shows "Record 10s"
- [ ] Click record → countdown starts from 10
- [ ] Recording auto-stops at 0
- [ ] File downloads as `iris_[timestamp].webm`
- [ ] Open file in VLC/browser
- [ ] **VERIFY: Watermark visible** (bottom-right):
  - "iRis • hologram studio"
  - "Created with CAVIAR"

### 3. Configure Stripe Test Mode
- [ ] Login to https://dashboard.stripe.com
- [ ] Switch to **Test mode** (toggle in header)
- [ ] Create Product: "iRis Plus" ($X/month)
- [ ] Create Product: "iRis Pro" ($Y/month)
- [ ] Copy Price IDs (format: `price_1234...`)
- [ ] Update `.env`:
```env
STRIPE_SECRET_KEY=sk_test_[your_key]
STRIPE_PRICE_PLUS=price_[plus_id]
STRIPE_PRICE_PRO=price_[pro_id]
```
- [ ] Restart dev server

### 4. Test Stripe Checkout
- [ ] Visit `/pricing`
- [ ] Three cards displayed (Free, Plus, Pro)
- [ ] Click "Get Plus"
- [ ] Stripe Checkout opens
- [ ] Enter test card: `4242 4242 4242 4242`
- [ ] Any future expiry (e.g., 12/34)
- [ ] Any CVC (e.g., 123)
- [ ] Any billing details
- [ ] Submit payment
- [ ] Redirected to `/thank-you?plan=plus`
- [ ] Message confirms upgrade

### 5. Test Plus Plan Recording
- [ ] Return to `/hologram` or `/hologram-studio`
- [ ] Recorder bar shows "Plus" pill
- [ ] Button shows "Record 60s"
- [ ] Start recording
- [ ] **NO WATERMARK** in preview
- [ ] Can record up to 60s
- [ ] Stop recording
- [ ] File downloads
- [ ] **VERIFY: No watermark** in video file

### 6. Test Pro Plan (Optional)
- [ ] Visit `/pricing`
- [ ] Click "Get Pro"
- [ ] Complete checkout with test card
- [ ] Return to hologram page
- [ ] Recorder shows "Pro" pill
- [ ] Button shows "Record 300s"
- [ ] **VERIFY: No watermark**

### 7. Webhook Testing (Optional)
```bash
# Terminal 1 - Keep dev server running

# Terminal 2 - Install Stripe CLI if needed
# https://stripe.com/docs/stripe-cli
stripe login
stripe listen --forward-to localhost:5173/api/billing/webhook

# Terminal 3 - Trigger test event
stripe trigger checkout.session.completed
```
- [ ] Webhook endpoint receives event
- [ ] Console logs "Stripe webhook event: checkout.session.completed"

## Troubleshooting

### Issue: "Cannot find module 'stripe'"
```bash
npm install stripe
```

### Issue: Watermark not appearing
- Check canvas has `id="holo-canvas"`
- Verify `needsWatermark()` returns true for free plan
- Check browser console for errors

### Issue: Recording fails
- Check browser supports MediaRecorder
- Try different MIME types
- Check canvas is rendering

### Issue: Stripe checkout not opening
- Verify API keys in `.env`
- Check browser console for errors
- Ensure prices exist in Stripe dashboard

### Issue: Plan not updating after payment
- Check localStorage in DevTools
- Verify redirect URL includes `?plan=` parameter
- Clear localStorage and retry

## Test Data Reference

### Test Cards
- Success: `4242 4242 4242 4242`
- Decline: `4000 0000 0000 0002`
- Auth required: `4000 0025 0000 3155`

### localStorage Keys
- Plan: `iris.plan` (values: 'free', 'plus', 'pro')

### File Naming
- Pattern: `iris_[timestamp].webm`
- Example: `iris_1735789456123.webm`

## Success Criteria
✅ All checkboxes marked
✅ Free tier has watermark
✅ Paid tiers have no watermark
✅ Recording limits enforced
✅ Stripe checkout completes
✅ Plan persists after payment

---
Phase 2 Testing Complete: ___/___/2025
Tested by: ________________