// ${IRIS_ROOT}\playwright\tests\iris_nowave.spec.ts
import { test, expect } from '@playwright/test';

test.describe('IRIS Wave Pipeline Exclusion', () => {
  test('IRIS loads without wave pipeline modules', async ({ page }) => {
    const logs: string[] = [];
    const errors: string[] = [];
    
    // Capture console logs
    page.on('console', msg => {
      const text = msg.text();
      logs.push(text);
      if (msg.type() === 'error') {
        errors.push(text);
      }
    });
    
    // Navigate to IRIS/standalone-holo
    await page.goto('/');
    
    // Wait for app to initialize
    await page.waitForTimeout(500);
    
    // Check that no wave-related modules were loaded
    const waveLoaded = logs.some(log => 
      /FftWaveBackend|AngularSpectrum|Gerchberg|FFT wave backend initialized/i.test(log)
    );
    
    // Verify null backend is used instead
    const nullBackendLoaded = logs.some(log => 
      /Wave processing disabled|using null backend/i.test(log)
    );
    
    // Assertions
    expect(waveLoaded).toBeFalsy();
    expect(nullBackendLoaded).toBeTruthy();
    
    // Ensure no errors related to missing wave modules
    const waveErrors = errors.filter(err => 
      /wave|fft|propagation|gerchberg/i.test(err)
    );
    expect(waveErrors).toHaveLength(0);
  });
  
  test('Wave backend returns no-op for render calls', async ({ page }) => {
    await page.goto('/');
    
    // Execute JavaScript in the page context to check wave backend
    const waveEnabled = await page.evaluate(() => {
      // Check if __IRIS_WAVE__ flag exists and is false
      return typeof (window as any).__IRIS_WAVE__ !== 'undefined' 
        ? (window as any).__IRIS_WAVE__ 
        : false;
    });
    
    expect(waveEnabled).toBeFalsy();
  });
  
  test('Bundle size check - no FFT compute shaders', async ({ page }) => {
    const resourceSizes: { url: string; size: number }[] = [];
    
    // Track all loaded resources
    page.on('response', response => {
      const url = response.url();
      const headers = response.headers();
      const size = parseInt(headers['content-length'] || '0', 10);
      
      if (url.includes('.js') || url.includes('.wgsl')) {
        resourceSizes.push({ url, size });
      }
    });
    
    await page.goto('/');
    await page.waitForTimeout(1000);
    
    // Check that no FFT/wave-specific shader files are loaded
    const waveShaders = resourceSizes.filter(r => 
      /fft.*\.wgsl|propagation.*\.wgsl|gerchberg.*\.wgsl|angular.*\.wgsl/i.test(r.url)
    );
    
    expect(waveShaders).toHaveLength(0);
    
    // Optional: Log total JS bundle size for monitoring
    const totalJsSize = resourceSizes
      .filter(r => r.url.endsWith('.js'))
      .reduce((sum, r) => sum + r.size, 0);
    
    console.log(`Total JS bundle size: ${(totalJsSize / 1024 / 1024).toFixed(2)} MB`);
    
    // Assert bundle is reasonably sized (adjust threshold as needed)
    expect(totalJsSize).toBeLessThan(5 * 1024 * 1024); // Less than 5MB
  });
});

test.describe('IRIS Labs Mode (when enabled)', () => {
  test.skip('Wave processing loads when VITE_IRIS_ENABLE_WAVE=1', async ({ page }) => {
    // This test would only run in Labs builds
    // Skipped by default in production
    
    // Would need to set up test with labs environment:
    // process.env.VITE_IRIS_ENABLE_WAVE = '1';
    
    const logs: string[] = [];
    page.on('console', msg => logs.push(msg.text()));
    
    await page.goto('/');
    await page.waitForTimeout(500);
    
    const waveLoaded = logs.some(log => 
      /FFT wave backend initialized|FftWaveBackend.*Ready/i.test(log)
    );
    
    expect(waveLoaded).toBeTruthy();
  });
});
