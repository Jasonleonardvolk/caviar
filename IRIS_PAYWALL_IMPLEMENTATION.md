# iRis Hologram Studio - Paywall & Recording Implementation

## Overview
Complete implementation of recording, plan gating, Stripe checkout, and pricing for the iRis hologram studio.

## Features Implemented

### 1. **Plan Tiers** (`/static/plans.json`)
- **Free**: 10s recording with watermark, WebM export only
- **Plus**: 60s recording, no watermark, WebM + MP4 export
- **Pro**: 5 min recording, no watermark, WebM + MP4 + ProRes export

### 2. **Recording System** (`HologramRecorder.svelte`)
- Composite canvas recording (watermark burned in)
- Plan-based duration limits
- Automatic watermark for Free tier
- Cross-browser compatible WebM recording
- High-quality 6Mbps video bitrate

### 3. **Stripe Integration**
- Checkout API endpoint (`/api/billing/checkout`)
- Webhook handler for payment confirmation
- Test-mode ready configuration
- Promotion code support enabled

### 4. **User Experience**
- Pricing page at `/pricing`
- Thank you page with plan activation
- Recording UI overlay with plan indicator
- Countdown timer during recording

## Quick Start

### 1. Install Dependencies
```bash
cd D:\Dev\kha\tori_ui_svelte
npm install stripe
```

Or run the setup script:
```powershell
.\setup-iris-paywall.ps1
```

### 2. Configure Stripe

1. Go to [Stripe Dashboard](https://dashboard.stripe.com)
2. Create test products:
   - **Plus Plan**: $X/month
   - **Pro Plan**: $Y/month
3. Copy the Price IDs (format: `price_xxxxx`)
4. Update `.env` file:
```env
STRIPE_SECRET_KEY=sk_test_your_actual_key_here
STRIPE_PRICE_PLUS=price_your_plus_price_id
STRIPE_PRICE_PRO=price_your_pro_price_id
```

### 3. Run Development Server
```bash
npm run dev
```

### 4. Test the Flow
1. Visit: http://localhost:5173/hologram-studio
2. Try recording (10s limit for Free)
3. Visit: http://localhost:5173/pricing
4. Click upgrade button (test mode)
5. Complete Stripe checkout
6. Return to app with upgraded plan

## File Structure Created

```
D:\Dev\kha\tori_ui_svelte\
├── static\
│   └── plans.json                    # Plan definitions
├── src\
│   ├── lib\
│   │   ├── stores\
│   │   │   └── userPlan.ts          # Plan state management
│   │   └── components\
│   │       └── HologramRecorder.svelte  # Recording component
│   └── routes\
│       ├── pricing\
│       │   └── +page.svelte          # Pricing page
│       ├── thank-you\
│       │   └── +page.svelte          # Post-payment page
│       ├── hologram-studio\
│       │   └── +page.svelte          # Demo studio page
│       └── api\
│           └── billing\
│               ├── checkout\
│               │   └── +server.ts    # Stripe checkout
│               └── webhook\
│                   └── +server.ts    # Payment webhook
└── .env                              # Configuration
```

## Testing Checklist

- [ ] Free tier: 10s recording with watermark
- [ ] Watermark appears correctly (bottom-right)
- [ ] Recording stops at time limit
- [ ] File downloads as WebM
- [ ] Pricing page displays all tiers
- [ ] Stripe checkout opens correctly
- [ ] Thank you page updates plan
- [ ] Plus/Pro unlocks longer recording
- [ ] No watermark on paid plans

## Production Deployment

### 1. Update Environment Variables
In Vercel/production:
```env
STRIPE_SECRET_KEY=sk_live_xxx
STRIPE_PRICE_PLUS=price_live_plus_id
STRIPE_PRICE_PRO=price_live_pro_id
STRIPE_SUCCESS_URL=https://yourdomain.com/thank-you
STRIPE_CANCEL_URL=https://yourdomain.com/pricing?canceled=1
```

### 2. Configure Stripe Webhook
1. In Stripe Dashboard → Webhooks
2. Add endpoint: `https://yourdomain.com/api/billing/webhook`
3. Select events: `checkout.session.completed`
4. Copy webhook secret to env

### 3. Implement User System
Replace localStorage with proper user database:
- Store subscription status in database
- Link Stripe customer ID to user
- Handle subscription lifecycle events

## Video Creation Guide

### Video 1: Quick Demo (30s)
- Show hologram rendering
- Click record button
- Show 10s countdown
- Download file with watermark
- "Upgrade for longer recordings"

### Video 2: Upgrade Flow (45s)
- Start on pricing page
- Click "Get Plus"
- Show Stripe checkout
- Complete payment (test card)
- Return to app
- Record 60s without watermark

### Video 3: Features Overview (60s)
- Show all three tiers
- Demonstrate watermark on/off
- Show export formats
- Recording duration differences
- Professional use cases

## Support & Next Steps

1. **Add export formats**: Implement MP4/ProRes converters
2. **Cloud storage**: Upload recordings to S3/GCS
3. **User dashboard**: View recording history
4. **Team plans**: Multi-user subscriptions
5. **Analytics**: Track conversion rates

## Troubleshooting

### "Stripe is not defined"
Run: `npm install stripe`

### Watermark not showing
Check canvas ID matches: `id="holo-canvas"`

### Recording fails
Ensure HTTPS in production (required for MediaRecorder)

### Payment not updating plan
Check webhook configuration and logs

---

Built with ❤️ for iRis Hologram Studio