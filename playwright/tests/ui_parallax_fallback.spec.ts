// ${IRIS_ROOT}\playwright\tests\ui_parallax_fallback.spec.ts
import { test, expect } from '@playwright/test';

const ROUTE = '/iris'; // change if your demo route differs

test.describe('Parallax quality baseline', () => {
  test('idle jitter < 0.3% of width (approx)', async ({ page }) => {
    await page.goto(ROUTE);
    const canvas = page.locator('canvas').first();
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox(); if (!box) test.fail(true, 'No canvas bounds');

    // center cursor
    await page.mouse.move(box.x + box.width/2, box.y + box.height/2);

    const samples: number[] = [];
    for (let i=0;i<120;i++) {
      const x = await page.evaluate(() => (window as any).__PARALLAX_DEBUG?.getPredictedPose()?.p[0] ?? 0);
      samples.push(x);
      await page.waitForTimeout(8); // ~120Hz
    }
    const mean = samples.reduce((a,b)=>a+b,0)/samples.length;
    const varr = samples.reduce((a,b)=>a+(b-mean)*(b-mean),0)/samples.length;
    const std = Math.sqrt(varr);

    // "screen units" assumed ~[0..1]; adjust if using pixels
    expect(std).toBeLessThan(0.003); // 0.3%
  });

  test('step overshoot < 5% of step magnitude', async ({ page }) => {
    await page.goto(ROUTE);
    const canvas = page.locator('canvas').first();
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox(); if (!box) test.fail(true, 'No canvas bounds');

    const leftX  = box.x + box.width * 0.10;
    const rightX = box.x + box.width * 0.90;
    const y      = box.y + box.height * 0.50;

    await page.mouse.move(leftX, y);
    await page.waitForTimeout(200);

    const x0 = await page.evaluate(() => (window as any).__PARALLAX_DEBUG?.getPredictedPose()?.p[0] ?? 0);

    await page.mouse.move(rightX, y);

    const path: number[] = [];
    for (let t=0;t<300;t+=8) {
      const x = await page.evaluate(() => (window as any).__PARALLAX_DEBUG?.getPredictedPose()?.p[0] ?? 0);
      path.push(x);
      await page.waitForTimeout(8);
    }

    const xFinal = path.slice(-10).reduce((a,b)=>a+b,0)/10;
    const maxX = Math.max(...path);
    const step = xFinal - x0;
    const overshoot = maxX - xFinal;
    const pct = step !== 0 ? Math.abs(overshoot / step) : 0;

    expect(pct).toBeLessThan(0.05);
  });
});
