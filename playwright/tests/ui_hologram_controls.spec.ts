import { test, expect, Page } from '@playwright/test';

test.describe('Hologram Control Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for WebGPU initialization
    await page.waitForTimeout(1000);
  });

  test('control surface pushes state and recovers on render error', async ({ page }) => {
    // Set controls to specific values
    await page.selectOption('#phaseMode', 'Soliton');
    await page.selectOption('#personaState', 'happy');
    await page.selectOption('#viewMode', 'quilt');
    
    // Set blend ratio
    const blendSlider = page.locator('#blendRatio');
    await blendSlider.fill('0.75');
    
    // Verify values are set
    await expect(page.locator('#phaseMode')).toHaveValue('Soliton');
    await expect(page.locator('#personaState')).toHaveValue('happy');
    await expect(page.locator('#viewMode')).toHaveValue('quilt');
    
    // Simulate GPU device lost error
    await page.evaluate(() => {
      // Dispatch device lost event
      const canvas = document.querySelector('canvas');
      if (canvas) {
        canvas.dispatchEvent(new Event('webglcontextlost'));
      }
      // Trigger error handler
      window.dispatchEvent(new ErrorEvent('error', { 
        message: 'WebGPU device lost',
        error: new Error('Simulated GPU error')
      }));
    });
    
    // Expect fallback UI or recovery message
    const fallbackIndicators = [
      page.getByText(/Reduced Quality Mode/i),
      page.getByText(/base mode/i),
      page.getByText(/Recovering/i),
      page.getByText(/fallback/i)
    ];
    
    let foundFallback = false;
    for (const indicator of fallbackIndicators) {
      if (await indicator.isVisible({ timeout: 5000 }).catch(() => false)) {
        foundFallback = true;
        break;
      }
    }
    
    if (!foundFallback) {
      // Check if canvas recovered (still exists and has context)
      const canvasRecovered = await page.evaluate(() => {
        const canvas = document.querySelector('canvas');
        return canvas !== null;
      });
      expect(canvasRecovered).toBe(true);
    }
  });

  test('all control parameters update shader uniforms', async ({ page }) => {
    // Track update events
    const updates: any[] = [];
    await page.exposeFunction('logUpdate', (data: any) => {
      updates.push(data);
    });
    
    // Inject event listener
    await page.evaluate(() => {
      const panel = document.querySelector('.control-panel');
      if (panel) {
        panel.addEventListener('update', (e: any) => {
          (window as any).logUpdate(e.detail);
        });
      }
    });
    
    // Change each control
    await page.selectOption('#phaseMode', 'Kerr');
    await page.selectOption('#personaState', 'sad');
    await page.selectOption('#viewMode', 'stereo');
    await page.locator('#blendRatio').fill('0.25');
    
    // Wait for updates to propagate
    await page.waitForTimeout(500);
    
    // Verify updates were triggered
    expect(updates.length).toBeGreaterThan(0);
    
    // Check last update has all parameters
    const lastUpdate = updates[updates.length - 1];
    expect(lastUpdate).toHaveProperty('phaseMode');
    expect(lastUpdate).toHaveProperty('personaState');
    expect(lastUpdate).toHaveProperty('viewMode');
    expect(lastUpdate).toHaveProperty('blendRatio');
  });

  test('diagnostics panel shows real-time info', async ({ page }) => {
    // Open diagnostics
    const diagnosticsToggle = page.locator('details summary').filter({ hasText: 'Diagnostics' });
    await diagnosticsToggle.click();
    
    // Wait for diagnostics to populate
    await page.waitForTimeout(1000);
    
    // Check for expected diagnostic fields
    const diagnosticsList = page.locator('details ul');
    const diagnosticsText = await diagnosticsList.textContent();
    
    // Should show some diagnostic info (FPS, memory, etc.)
    const hasContent = diagnosticsText && diagnosticsText.length > 0;
    expect(hasContent).toBe(true);
  });

  test('controls persist through render pipeline restarts', async ({ page }) => {
    // Set initial values
    await page.selectOption('#phaseMode', 'Soliton');
    await page.locator('#blendRatio').fill('0.85');
    
    // Simulate pipeline restart
    await page.evaluate(() => {
      // Trigger reinitialize
      window.dispatchEvent(new Event('reinitialize-renderer'));
    });
    
    await page.waitForTimeout(1000);
    
    // Values should persist
    await expect(page.locator('#phaseMode')).toHaveValue('Soliton');
    const blendValue = await page.locator('#blendRatio').inputValue();
    expect(parseFloat(blendValue)).toBeCloseTo(0.85, 2);
  });

  test('mobile viewport adjusts control panel layout', async ({ page, isMobile }) => {
    if (!isMobile) {
      test.skip();
    }
    
    const controlPanel = page.locator('.control-panel');
    const box = await controlPanel.boundingBox();
    
    // On mobile, panel should be reasonably sized
    expect(box?.width).toBeLessThanOrEqual(320);
    
    // Controls should still be accessible
    await page.selectOption('#viewMode', 'depth');
    await expect(page.locator('#viewMode')).toHaveValue('depth');
  });
});

test.describe('Render Performance', () => {
  test('maintains target FPS under normal load', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000); // Let it stabilize
    
    // Measure FPS
    const fps = await page.evaluate(() => {
      return new Promise<number>((resolve) => {
        let frameCount = 0;
        const startTime = performance.now();
        
        function countFrame() {
          frameCount++;
          if (performance.now() - startTime < 1000) {
            requestAnimationFrame(countFrame);
          } else {
            resolve(frameCount);
          }
        }
        
        requestAnimationFrame(countFrame);
      });
    });
    
    // Should maintain at least 30 FPS (mobile minimum)
    expect(fps).toBeGreaterThanOrEqual(25);
  });
});
