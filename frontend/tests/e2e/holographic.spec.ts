import { test, expect } from '@playwright/test';

/**
 * E2E tests for the TORI Holographic Display System
 */

test.describe('Holographic Display System', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load the application', async ({ page }) => {
    // Check page title
    await expect(page).toHaveTitle(/TORI Holographic/);
    
    // Check main container exists
    const mainContainer = page.locator('#app');
    await expect(mainContainer).toBeVisible();
  });

  test('should check WebGPU availability', async ({ page }) => {
    // Check if WebGPU is available
    const hasWebGPU = await page.evaluate(() => {
      return 'gpu' in navigator;
    });
    
    if (hasWebGPU) {
      // WebGPU canvas should be visible
      const canvas = page.locator('canvas#webgpu-canvas');
      await expect(canvas).toBeVisible();
    } else {
      // Fallback message should be shown
      const fallback = page.locator('.webgl-fallback');
      await expect(fallback).toBeVisible();
    }
  });

  test('should load quilt manifest', async ({ page }) => {
    // Navigate to quilt viewer
    await page.goto('/viewer');
    
    // Wait for manifest to load
    await page.waitForResponse(response => 
      response.url().includes('/assets/quilt/manifest.json') && 
      response.status() === 200
    );
    
    // Check quilt selector exists
    const quiltSelector = page.locator('#quilt-selector');
    await expect(quiltSelector).toBeVisible();
    
    // Should have at least one quilt option
    const options = await quiltSelector.locator('option').count();
    expect(options).toBeGreaterThan(0);
  });

  test('should switch rendering modes', async ({ page }) => {
    await page.goto('/viewer');
    
    // Wait for renderer to initialize
    await page.waitForSelector('#render-mode-selector');
    
    const modeSelector = page.locator('#render-mode-selector');
    
    // Test each rendering mode
    const modes = ['standard', 'holographic', 'lightfield', 'volumetric', 'depth'];
    
    for (const mode of modes) {
      await modeSelector.selectOption(mode);
      
      // Check mode is active
      const activeMode = await page.evaluate(() => {
        return window.renderer?.getActiveMode();
      });
      
      expect(activeMode).toBe(mode);
      
      // Canvas should still be rendering
      const canvas = page.locator('canvas#webgpu-canvas');
      await expect(canvas).toBeVisible();
    }
  });

  test('should handle texture loading', async ({ page }) => {
    await page.goto('/viewer');
    
    // Monitor texture loading
    const textureLoadPromise = page.waitForEvent('console', msg => 
      msg.text().includes('Texture loaded')
    );
    
    // Select a quilt
    const quiltSelector = page.locator('#quilt-selector');
    await quiltSelector.selectOption('demo_5x9');
    
    // Wait for texture load message
    await textureLoadPromise;
    
    // Check texture is displayed
    const canvas = page.locator('canvas#webgpu-canvas');
    const screenshot = await canvas.screenshot();
    expect(screenshot).toBeTruthy();
  });

  test('should register service worker', async ({ page }) => {
    // Check service worker registration
    const hasServiceWorker = await page.evaluate(async () => {
      if ('serviceWorker' in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        return registration !== undefined;
      }
      return false;
    });
    
    expect(hasServiceWorker).toBe(true);
    
    // Check cache is populated
    const cacheExists = await page.evaluate(async () => {
      if ('caches' in window) {
        const cacheNames = await caches.keys();
        return cacheNames.some(name => name.startsWith('tori-'));
      }
      return false;
    });
    
    expect(cacheExists).toBe(true);
  });

  test('should handle head tracking', async ({ page, context }) => {
    // Grant permissions for device orientation
    await context.grantPermissions(['accelerometer', 'gyroscope']);
    
    await page.goto('/viewer');
    
    // Enable head tracking
    const trackingToggle = page.locator('#head-tracking-toggle');
    await trackingToggle.click();
    
    // Simulate device orientation change
    await page.evaluate(() => {
      const event = new DeviceOrientationEvent('deviceorientation', {
        alpha: 45,
        beta: 30,
        gamma: 15
      });
      window.dispatchEvent(event);
    });
    
    // Check if view matrix updated
    const viewUpdated = await page.evaluate(() => {
      return window.renderer?.viewMatrixUpdated === true;
    });
    
    expect(viewUpdated).toBe(true);
  });

  test('should handle video playback', async ({ page }) => {
    await page.goto('/viewer');
    
    // Load video quilt
    const videoButton = page.locator('#load-video-quilt');
    await videoButton.click();
    
    // Wait for video to start
    await page.waitForFunction(() => {
      const video = document.querySelector('video');
      return video && video.readyState >= 2;
    });
    
    // Check playback controls
    const playButton = page.locator('#play-button');
    const pauseButton = page.locator('#pause-button');
    
    await pauseButton.click();
    const isPaused = await page.evaluate(() => {
      const video = document.querySelector('video');
      return video?.paused;
    });
    expect(isPaused).toBe(true);
    
    await playButton.click();
    const isPlaying = await page.evaluate(() => {
      const video = document.querySelector('video');
      return video && !video.paused;
    });
    expect(isPlaying).toBe(true);
  });

  test('should measure performance', async ({ page }) => {
    await page.goto('/viewer');
    
    // Wait for renderer to stabilize
    await page.waitForTimeout(2000);
    
    // Get performance metrics
    const metrics = await page.evaluate(() => {
      return {
        fps: window.renderer?.getCurrentFPS(),
        frameTime: window.renderer?.getFrameTime(),
        gpuTime: window.renderer?.getGPUTime(),
        textureMemory: window.renderer?.getTextureMemoryUsage()
      };
    });
    
    // Check performance thresholds
    expect(metrics.fps).toBeGreaterThanOrEqual(30); // Minimum 30 FPS
    expect(metrics.frameTime).toBeLessThanOrEqual(33); // Max 33ms per frame
    
    console.log('Performance metrics:', metrics);
  });

  test('should handle errors gracefully', async ({ page }) => {
    // Listen for error events
    const errors: string[] = [];
    page.on('pageerror', error => {
      errors.push(error.message);
    });
    
    await page.goto('/viewer');
    
    // Try to load non-existent quilt
    await page.evaluate(() => {
      window.renderer?.loadQuilt('non-existent-quilt');
    });
    
    // Should show error message, not crash
    const errorMessage = page.locator('.error-message');
    await expect(errorMessage).toBeVisible();
    
    // No uncaught errors
    expect(errors.length).toBe(0);
  });
});

test.describe('Mobile Support', () => {
  test.use({ 
    viewport: { width: 375, height: 667 },
    userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15'
  });

  test('should adapt to mobile viewport', async ({ page }) => {
    await page.goto('/');
    
    // Check mobile menu is visible
    const mobileMenu = page.locator('.mobile-menu');
    await expect(mobileMenu).toBeVisible();
    
    // Check touch controls are enabled
    const hasTouchControls = await page.evaluate(() => {
      return window.renderer?.touchControlsEnabled === true;
    });
    expect(hasTouchControls).toBe(true);
  });

  test('should handle touch gestures', async ({ page }) => {
    await page.goto('/viewer');
    
    const canvas = page.locator('canvas#webgpu-canvas');
    
    // Simulate pinch zoom
    await canvas.dispatchEvent('touchstart', {
      touches: [
        { clientX: 100, clientY: 100 },
        { clientX: 200, clientY: 200 }
      ]
    });
    
    await canvas.dispatchEvent('touchmove', {
      touches: [
        { clientX: 50, clientY: 50 },
        { clientX: 250, clientY: 250 }
      ]
    });
    
    await canvas.dispatchEvent('touchend');
    
    // Check zoom level changed
    const zoomLevel = await page.evaluate(() => {
      return window.renderer?.getZoomLevel();
    });
    
    expect(zoomLevel).not.toBe(1);
  });
});

test.describe('Accessibility', () => {
  test('should have proper ARIA labels', async ({ page }) => {
    await page.goto('/');
    
    // Check main navigation
    const nav = page.locator('nav[role="navigation"]');
    await expect(nav).toHaveAttribute('aria-label', 'Main navigation');
    
    // Check buttons have labels
    const buttons = page.locator('button');
    const count = await buttons.count();
    
    for (let i = 0; i < count; i++) {
      const button = buttons.nth(i);
      const ariaLabel = await button.getAttribute('aria-label');
      const text = await button.textContent();
      
      expect(ariaLabel || text).toBeTruthy();
    }
  });

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/');
    
    // Tab through interactive elements
    await page.keyboard.press('Tab');
    const firstFocused = await page.evaluate(() => document.activeElement?.tagName);
    expect(firstFocused).toBeTruthy();
    
    // Navigate with arrow keys
    await page.keyboard.press('ArrowDown');
    const secondFocused = await page.evaluate(() => document.activeElement?.tagName);
    expect(secondFocused).toBeTruthy();
    
    // Activate with Enter
    await page.keyboard.press('Enter');
    // Should trigger action without errors
  });

  test('should support screen readers', async ({ page }) => {
    await page.goto('/');
    
    // Check for screen reader announcements
    const liveRegion = page.locator('[aria-live]');
    await expect(liveRegion).toHaveCount(1);
    
    // Trigger an action that should announce
    const button = page.locator('#load-quilt-button');
    await button.click();
    
    // Check announcement was made
    const announcement = await liveRegion.textContent();
    expect(announcement).toContain('Loading');
  });
});
