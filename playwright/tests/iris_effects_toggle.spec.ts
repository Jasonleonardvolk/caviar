// ${IRIS_ROOT}\playwright\tests\iris_effects_toggle.spec.ts
import { test, expect } from '@playwright/test';

const ROUTE = '/iris'; // adjust if your demo route differs
const LS_KEY = 'iris_settings_v1';

async function primeSettings(page, obj: Record<string, any>) {
  await page.addInitScript(([k, v]) => {
    try {
      localStorage.clear();
      localStorage.setItem(k, JSON.stringify(v));
    } catch {}
  }, [LS_KEY, obj]);
}

function captureConsole(page) {
  const logs: string[] = [];
  page.on('console', m => logs.push(m.text()));
  return logs;
}

test.describe('IRIS perceptual effects toggles', () => {

  test('CA on, DOF off → only ca dispatches', async ({ page }) => {
    const logs = captureConsole(page);
    await primeSettings(page, {
      // minimize noise: disable everything except CA
      quality: 'standard',
      jbuEnabled: false,
      caEnabled: true,   caStrength: 0.12,
      dofEnabled: false,
      mblurEnabled: false
    });

    await page.goto(ROUTE);
    await page.waitForTimeout(300); // let first frame(s) run

    const sawCA   = logs.some(s => s.includes('[fx] ca:dispatch'));
    const sawDOF  = logs.some(s => s.includes('[fx] dof:dispatch'));
    const sawMB   = logs.some(s => s.includes('[fx] mblur:dispatch'));

    expect(sawCA).toBeTruthy();
    expect(sawDOF).toBeFalsy();
    expect(sawMB).toBeFalsy();
  });

  test('DOF on, CA off → only dof dispatches', async ({ page }) => {
    const logs = captureConsole(page);
    await primeSettings(page, {
      quality: 'standard',
      jbuEnabled: false,
      caEnabled: false,
      dofEnabled: true,  dofStrength: 0.2,
      mblurEnabled: false
    });

    await page.goto(ROUTE);
    await page.waitForTimeout(300);

    const sawDOF = logs.some(s => s.includes('[fx] dof:dispatch'));
    const sawCA  = logs.some(s => s.includes('[fx] ca:dispatch'));
    const sawMB  = logs.some(s => s.includes('[fx] mblur:dispatch'));

    expect(sawDOF).toBeTruthy();
    expect(sawCA).toBeFalsy();
    expect(sawMB).toBeFalsy();
  });

  test('JBU default on → jbu dispatches (depth upsample smoke)', async ({ page }) => {
    const logs = captureConsole(page);
    await primeSettings(page, {
      // use defaults: JBU is on by default in our settings
      quality: 'standard',
      jbuEnabled: true,
      caEnabled: false,
      dofEnabled: false,
      mblurEnabled: false
    });

    await page.goto(ROUTE);
    await page.waitForTimeout(300);

    const sawJBU = logs.some(s => s.includes('[fx] jbu:dispatch'));
    expect(sawJBU).toBeTruthy();
  });

  test('Motion blend on → history blend dispatches', async ({ page }) => {
    const logs = captureConsole(page);
    await primeSettings(page, {
      quality: 'standard',
      jbuEnabled: false,
      caEnabled: false,
      dofEnabled: false,
      mblurEnabled: true, mblurStrength: 0.15
    });

    await page.goto(ROUTE);
    await page.waitForTimeout(500);

    const sawMB = logs.some(s => s.includes('[fx] mblur:dispatch'));
    expect(sawMB).toBeTruthy();
  });

});
