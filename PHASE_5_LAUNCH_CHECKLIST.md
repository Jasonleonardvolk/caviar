# PHASE 5 - LAUNCH DAY CHECKLIST
# iRis Minimum Viable Blast

## Pre-Launch Verification (30 mins)

### Technical Check
- [ ] Dev server runs: `pnpm dev`
- [ ] Hologram studio loads at `/hologram-studio`
- [ ] Free recording works (10s + watermark)
- [ ] Pricing page loads at `/pricing`
- [ ] Stripe checkout opens (test mode)
- [ ] Plus recording works (60s, no watermark)
- [ ] Videos download correctly
- [ ] Vercel deployment live

### Content Check
- [ ] Landing page updated with new copy
- [ ] 3 marketing videos ready (A, B, C)
- [ ] Social posts drafted
- [ ] #HologramDrop page created
- [ ] Founders-100 terms ready

## Launch Sequence (2 hours)

### Hour 1: Deploy & Test

#### 1. Update Landing Page (15 mins)
```bash
# Replace landing page
cp src/routes/+page.svelte.new src/routes/+page.svelte

# Commit and push
git add .
git commit -m "Launch: New landing page with CTAs"
git push origin main
```

#### 2. Final Production Test (15 mins)
- [ ] Visit production URL
- [ ] Test Free recording
- [ ] Test Stripe checkout
- [ ] Test Plus recording
- [ ] Verify on mobile

#### 3. Upload Videos (15 mins)
- [ ] Upload to YouTube/Vimeo (unlisted)
- [ ] Upload to Twitter Media Studio
- [ ] Upload to Instagram
- [ ] Copy links for posts

#### 4. Prepare Social Accounts (15 mins)
- [ ] Update bio links
- [ ] Pin launch tweet
- [ ] Set profile images
- [ ] Add story highlights

### Hour 2: Launch Blast

#### 5. Social Media Posts (30 mins)

**X/Twitter:**
```
We killed the 45° glass. You can't mix two light fields without optics—but you can make footage that looks like you did.

iRis: Creator-Pro Hologram Studio
Record in minutes. Own the master.

🚀 #HologramDrop challenge LIVE
🎁 First 100 get Founders pricing

Try free → [URL]
```

**Instagram:**
```
Make 'hologram' videos brands pay for 🔮

✨ Physics-native looks
⚡ 10s free exports  
🚀 No watermark with Plus

Join #HologramDrop - win 1 year Pro!

Link in bio 👆
```

**TikTok:**
```
POV: You just made a sponsor-ready hologram in 60 seconds

Free 10s export TODAY only 
#HologramDrop #CreatorTools #HologramStudio
```

**LinkedIn:**
```
Launching iRis: Professional Hologram Studio for Creators

After months of development, we're excited to introduce iRis - a tool that lets creators produce physics-native holographic content without expensive hardware.

Key features:
• Free 10s exports to test
• Plus: 60s recordings, no watermark
• Pro: 5min recordings, all formats

Special launch offer: First 100 subscribers get lifetime Founders pricing.

Try it now: [URL]
```

#### 6. Direct Outreach (15 mins)
- [ ] DM 10 creator friends
- [ ] Post in 3 relevant Discord/Slack
- [ ] Email list (if available)
- [ ] WhatsApp key contacts

#### 7. Monitor & Engage (15 mins)
- [ ] Reply to comments
- [ ] RT/share user content
- [ ] Thank early adopters
- [ ] Fix any reported issues

## Money Triggers

### Immediate CTAs
1. **Landing**: "Try Free" → `/hologram-studio`
2. **After Free Record**: "Remove watermark" → `/pricing`
3. **Pricing Page**: "Get Plus $9.99" → Stripe
4. **Thank You**: "Start Recording" → `/hologram-studio`

### Pricing Psychology
- Free: Creates urgency (watermark)
- Plus ($9.99): Impulse buy threshold
- Pro ($29.99): Anchors value
- Founders-100: FOMO trigger

### Conversion Path
```
Land → Try Free (10s) → See watermark → Want removal → Buy Plus → Record 60s → Share → Viral
```

## Crisis Management

### If Something Breaks

#### Recording not working:
```javascript
// Emergency fix in console
localStorage.setItem('iris.plan', 'plus');
location.reload();
```

#### Stripe failing:
- Switch to manual PayPal/Venmo
- DM for payment
- Honor later with manual upgrade

#### Site down:
- Post videos directly
- Link to Google Form for signups
- "Site hugged to death! DM for early access"

### If No Sales:
1. **Hour 3**: Add "50% off TODAY" banner
2. **Hour 6**: Launch with Plus at $4.99
3. **Day 2**: Free Plus for first 10 users

### If Too Many Sales:
1. Close Founders-100 early
2. Raise prices immediately
3. Add "Waitlist" for stability

## Success Metrics

### Launch Day Goals
- [ ] 100 free trials
- [ ] 10 Plus subscriptions
- [ ] 1 Pro subscription
- [ ] 50 #HologramDrop posts
- [ ] 1000 site visits

### Tracking
```javascript
// Add to landing page
gtag('event', 'conversion', {
  'send_to': 'AW-XXXXXX/XXXXX',
  'value': 9.99,
  'currency': 'USD'
});
```

## Post-Launch (Next 24h)

### Hour 3-6
- [ ] Respond to all DMs
- [ ] Share user creations
- [ ] Post tutorial thread
- [ ] Fix reported bugs

### Hour 6-12
- [ ] Send follow-up emails
- [ ] Create FAQ from questions
- [ ] Post in more communities
- [ ] Schedule tweets

### Hour 12-24
- [ ] Compile metrics
- [ ] Plan Day 2 push
- [ ] Prep customer support
- [ ] Document lessons

## Launch Commands

```bash
# Quick deploy
git add . && git commit -m "Launch" && git push

# Monitor logs
vercel logs --follow

# Quick stats
curl https://your-app.vercel.app/api/stats

# Emergency rollback
vercel rollback
```

## Final Checklist

MUST HAVE:
- [x] Recording works
- [x] Payment works
- [x] Videos ready
- [x] Social accounts ready

NICE TO HAVE:
- [ ] Email sequence
- [ ] Affiliate program
- [ ] Press release
- [ ] Influencer outreach

## GO LIVE

When ready:
1. Take a deep breath
2. Post to X/Twitter first
3. Cross-post immediately
4. Engage for 2 hours straight
5. Celebrate first sale! 🎉

---

**You got this! Ship it! 🚀**