# TORI Svelte Accessibility (A11y) Fixes Guide

## ✅ Fixes Applied

### 1. PersonaPanel.svelte
- **Fixed**: Changed overlay from `role="button"` to proper `role="dialog"` with `aria-modal="true"`
- **Fixed**: Added proper `aria-labelledby` to connect dialog with its title
- **Fixed**: Removed nested dialog roles (only one dialog role per modal)
- **Benefit**: Screen readers now properly announce the modal dialog

### 2. HolographicDisplay.svelte
- **Fixed**: Added `<track>` element for captions to the video element
- **Fixed**: Added `muted` and `playsinline` attributes for better browser compatibility
- **Created**: `/static/captions/hologram-video.vtt` placeholder file
- **Benefit**: WCAG compliance for video elements

## 📋 Common A11y Patterns to Fix

### 1. Clickable Divs → Use Buttons
```svelte
<!-- ❌ Bad -->
<div on:click={handleClick}>Click me</div>

<!-- ✅ Good -->
<button on:click={handleClick}>Click me</button>

<!-- ✅ Or if you must use div -->
<div 
  role="button"
  tabindex="0"
  on:click={handleClick}
  on:keydown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  }}
>
  Click me
</div>
```

### 2. Form Labels
```svelte
<!-- ❌ Bad -->
<input type="text" placeholder="Name" />

<!-- ✅ Good -->
<label for="user-name">Name:</label>
<input id="user-name" type="text" />

<!-- ✅ Or wrap the input -->
<label>
  Name:
  <input type="text" />
</label>
```

### 3. Interactive Elements Need Keyboard Support
```svelte
<!-- ❌ Bad -->
<div on:click={toggle}>Toggle</div>

<!-- ✅ Good -->
<button on:click={toggle}>Toggle</button>

<!-- ✅ Or add keyboard support -->
<div
  role="button"
  tabindex="0"
  on:click={toggle}
  on:keypress={(e) => e.key === 'Enter' && toggle()}
>
  Toggle
</div>
```

### 4. Videos Need Captions
```svelte
<!-- ❌ Bad -->
<video src="video.mp4"></video>

<!-- ✅ Good -->
<video src="video.mp4">
  <track kind="captions" src="/captions/video.vtt" srclang="en" label="English" />
</video>
```

### 5. Images Need Alt Text
```svelte
<!-- ❌ Bad -->
<img src="logo.png" />

<!-- ✅ Good (informative image) -->
<img src="logo.png" alt="TORI System Logo" />

<!-- ✅ Good (decorative image) -->
<img src="decoration.png" alt="" role="presentation" />
```

## 🔍 How to Find A11y Issues

1. **Run Svelte with a11y warnings enabled** (default in dev mode)
2. **Use browser DevTools**:
   - Chrome: Lighthouse audit
   - Firefox: Accessibility Inspector
3. **Test with keyboard only** - Tab through your app
4. **Use screen reader** - NVDA (Windows) or VoiceOver (Mac)

## 🎯 Quick Checklist

- [ ] All interactive elements are keyboard accessible
- [ ] All form inputs have labels
- [ ] All images have alt text
- [ ] Videos have captions
- [ ] Modals trap focus properly
- [ ] Color contrast meets WCAG standards (4.5:1 for normal text)
- [ ] Page has proper heading hierarchy (h1 → h2 → h3)
- [ ] ARIA labels for icon-only buttons

## 🚀 Testing Your Fixes

After making a11y fixes:

```bash
# The dev server will show warnings
npm run dev

# Run a production build to see if warnings persist
npm run build
```

## 📚 Resources

- [Svelte Accessibility Warnings](https://svelte.dev/docs#compile-time-svelte-compile)
- [WCAG Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [ARIA Authoring Practices](https://www.w3.org/WAI/ARIA/apg/)
