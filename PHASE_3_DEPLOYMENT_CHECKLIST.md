# PHASE 3 - DEPLOYMENT CHECKLIST
# iRis Hologram Studio → Production

## Pre-Deployment Checklist

### 1. Local Verification
- [ ] Phase 2 tests passing (Free/Plus/Pro recording)
- [ ] Watermark appears only on Free tier
- [ ] Stripe test checkout working
- [ ] Recording downloads working

### 2. Code Preparation
- [ ] Remove any debug console.log statements
- [ ] Verify all environment variables in code use process.env
- [ ] Check no hardcoded API keys or secrets
- [ ] Ensure error handling in place

## Deployment Steps

### Step 1: Install Vercel Adapter
```bash
cd D:\Dev\kha\tori_ui_svelte
npm install -D @sveltejs/adapter-vercel
```

### Step 2: Update svelte.config.js
Replace current config with Vercel adapter:
```javascript
import adapterVercel from '@sveltejs/adapter-vercel';
// ... rest of config
adapter: adapterVercel()
```
Or use the prepared `svelte.config.vercel.js`

### Step 3: Commit to GitHub
```bash
git add .
git commit -m "iRis launch: recorder + pricing + stripe checkout"
git push origin main
```

### Step 4: Deploy to Vercel

1. **Create Vercel Account**
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub

2. **Import Project**
   - Click "Add New Project"
   - Import `Jasonleonardvolk/caviar`
   - Select `tori_ui_svelte` as root directory

3. **Configure Build Settings**
   - Framework Preset: **SvelteKit**
   - Build Command: `pnpm build` or `npm run build`
   - Output Directory: `.svelte-kit`
   - Install Command: `pnpm install` or `npm install`

4. **Environment Variables (Production)**
   ```
   STRIPE_SECRET_KEY=sk_test_xxx (use sk_live_xxx for production)
   STRIPE_PRICE_PLUS=price_xxx
   STRIPE_PRICE_PRO=price_xxx
   STRIPE_SUCCESS_URL=https://your-app.vercel.app/thank-you
   STRIPE_CANCEL_URL=https://your-app.vercel.app/pricing?canceled=1
   VITE_API_URL=https://your-backend-api.com
   PUBLIC_BASE_URL=https://your-app.vercel.app
   ```

5. **Deploy**
   - Click "Deploy"
   - Wait for build to complete (~2-3 minutes)

### Step 5: Post-Deployment Testing

#### Smoke Test Checklist
- [ ] Visit: `https://your-app.vercel.app`
- [ ] Navigate to `/hologram-studio`
- [ ] Test Free recording (10s, watermark)
- [ ] Visit `/pricing`
- [ ] Test Stripe checkout (test mode)
- [ ] Verify redirect to `/thank-you`
- [ ] Test Plus recording (60s, no watermark)

#### API Endpoints
- [ ] Test: `https://your-app.vercel.app/api/billing/checkout` (POST)
- [ ] Test: `https://your-app.vercel.app/api/billing/webhook` (POST)

### Step 6: Configure Stripe Webhook (Production)

1. Go to Stripe Dashboard → Webhooks
2. Add endpoint: `https://your-app.vercel.app/api/billing/webhook`
3. Select events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
4. Copy signing secret
5. Add to Vercel env: `STRIPE_WEBHOOK_SECRET=whsec_xxx`

### Step 7: Switch to Live Mode (When Ready)

1. **Create Live Products in Stripe**
   - Switch to Live mode in Stripe Dashboard
   - Create Plus and Pro products with live prices
   - Copy live Price IDs

2. **Update Vercel Environment Variables**
   ```
   STRIPE_SECRET_KEY=sk_live_xxx (LIVE key)
   STRIPE_PRICE_PLUS=price_live_plus_xxx
   STRIPE_PRICE_PRO=price_live_pro_xxx
   ```

3. **Redeploy**
   - Vercel will auto-redeploy on env change
   - Or trigger manual redeploy

## Troubleshooting

### Build Fails
- Check `package.json` has all dependencies
- Verify `svelte.config.js` uses correct adapter
- Check build logs in Vercel dashboard

### Stripe Not Working
- Verify environment variables are set
- Check API routes are accessible
- Review function logs in Vercel

### Recording Not Working
- HTTPS required for MediaRecorder
- Check canvas element exists
- Verify browser compatibility

### Watermark Issues
- Clear browser cache
- Check localStorage for plan
- Verify plan detection logic

## Monitoring

### Vercel Dashboard
- Monitor build status
- Check function logs
- Review analytics

### Stripe Dashboard
- Monitor successful payments
- Check failed payments
- Review webhook events

## Rollback Plan

If issues arise:
1. Revert to previous deployment in Vercel
2. Or push previous commit:
   ```bash
   git revert HEAD
   git push origin main
   ```

---

## Success Criteria

✅ Site accessible at production URL
✅ Free tier recording with watermark works
✅ Stripe checkout completes successfully
✅ Plus/Pro tiers unlock features
✅ No console errors in production
✅ Mobile responsive

---

Deployment Date: ___________
Deployed By: _______________
Production URL: ____________