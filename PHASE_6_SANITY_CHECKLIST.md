# PHASE 6 - SANITY CHECKLIST
# Final verification before going live

## Core Functionality

### Free Tier
- [ ] Navigate to `/hologram-studio`
- [ ] Recorder shows "Free" badge
- [ ] Click "Record 10s"
- [ ] Recording stops at exactly 10 seconds
- [ ] File downloads as `iris_*.webm`
- [ ] **WATERMARK VISIBLE** in bottom-right
  - Text: "iRis • hologram studio"
  - Text: "Created with CAVIAR"

### Upgrade Flow
- [ ] Visit `/pricing`
- [ ] Three tiers displayed clearly
- [ ] Click "Get Plus" button
- [ ] Stripe checkout opens
- [ ] Test card: `4242 4242 4242 4242`
- [ ] Complete payment
- [ ] Redirect to `/thank-you?plan=plus`
- [ ] Success message displayed

### Plus Tier
- [ ] Return to `/hologram-studio`
- [ ] Recorder shows "Plus" badge
- [ ] Click "Record 60s"
- [ ] Can record up to 60 seconds
- [ ] Stop recording manually
- [ ] File downloads
- [ ] **NO WATERMARK** in video

### File Management
- [ ] Videos save to Downloads folder
- [ ] Files named `iris_[timestamp].webm`
- [ ] Can convert to MP4 using tools
- [ ] Marketing videos in `site\showcase\`
- [ ] Export demos in `exports\video\`

## Production Deployment

### Vercel Status
- [ ] Site accessible at production URL
- [ ] HTTPS working correctly
- [ ] No console errors
- [ ] Mobile responsive
- [ ] Fast load times (<3s)

### Cross-Browser Testing
- [ ] Chrome/Edge: Recording works
- [ ] Firefox: Recording works
- [ ] Safari: Basic functionality
- [ ] Mobile Chrome: Can view
- [ ] Mobile Safari: Can view

### API Endpoints
- [ ] `/api/billing/checkout` responds
- [ ] `/api/billing/webhook` responds
- [ ] Stripe keys configured
- [ ] Environment variables set

## Content & Marketing

### Landing Page
- [ ] Hero text compelling
- [ ] CTAs prominent
- [ ] Features explained
- [ ] Social proof added
- [ ] Mobile optimized

### Videos Ready
- [ ] A_shock_proof.mp4 (10s, watermark)
- [ ] B_how_to_60s.mp4 (20-30s tutorial)
- [ ] C_buyers_clip.mp4 (15-20s exports)
- [ ] Uploaded to cloud backup
- [ ] Compressed for social media

### Social Media
- [ ] X/Twitter post drafted
- [ ] Instagram post ready
- [ ] TikTok video prepared
- [ ] LinkedIn update written
- [ ] Hashtags selected

### Legal/Terms
- [ ] #HologramDrop rules posted
- [ ] Founders-100 terms ready
- [ ] Privacy policy linked
- [ ] Terms of service linked

## Money Flow

### Pricing Psychology
- [ ] Free creates urgency (watermark)
- [ ] Plus at impulse price ($9.99)
- [ ] Pro anchors value ($29.99)
- [ ] Founders-100 creates FOMO

### Conversion Path
```
Land → Try Free → See Watermark → Want Removal → Buy Plus → Happy Customer
```

### Payment Testing
- [ ] Stripe test mode working
- [ ] Test payment completes
- [ ] Plan updates after payment
- [ ] Ready to switch to live keys

## Crisis Preparedness

### Backup Plans
- [ ] Manual payment method ready
- [ ] Support email configured
- [ ] Bug report form ready
- [ ] Rollback plan documented

### Common Issues & Fixes

**Recording not working:**
```javascript
localStorage.setItem('iris.plan', 'plus');
location.reload();
```

**Stripe failing:**
- Use PayPal backup
- Manual upgrade via DM

**Site overloaded:**
- Scale Vercel dynos
- Enable CDN caching
- Post "overwhelmed by demand"

## Launch Metrics

### Success Indicators
- [ ] First visitor within 1 hour
- [ ] First free trial within 2 hours
- [ ] First paid customer within 6 hours
- [ ] 10+ social shares day one
- [ ] No critical bugs reported

### Tracking Setup
- [ ] Google Analytics configured
- [ ] Conversion tracking ready
- [ ] Stripe webhook logging
- [ ] Error monitoring active

## Final Pre-Launch

### 30 Minutes Before
- [ ] Clear browser cache
- [ ] Test entire flow once more
- [ ] Check all links work
- [ ] Verify videos play
- [ ] Deep breath

### 5 Minutes Before
- [ ] Open Stripe dashboard
- [ ] Open Vercel logs
- [ ] Open social media
- [ ] Prepare first post
- [ ] Hit send!

## Post-Launch Monitoring

### First Hour
- [ ] Watch for errors
- [ ] Respond to comments
- [ ] Share user content
- [ ] Fix urgent issues
- [ ] Celebrate first sale!

### First Day
- [ ] Compile feedback
- [ ] Address top issues
- [ ] Thank supporters
- [ ] Plan day 2 push
- [ ] Get some rest

---

## FINAL CHECK

**MUST WORK:**
- [x] Free recording with watermark
- [x] Stripe checkout
- [x] Plus recording without watermark
- [x] File downloads

**MUST HAVE:**
- [x] Landing page live
- [x] Videos ready
- [x] Social posts prepared
- [x] Payment processing

**SHIP IT:** ✅

---

*Remember: Done is better than perfect. You can iterate after launch!*