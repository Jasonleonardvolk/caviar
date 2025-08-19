// Run with: npx playwright test frontend/hybrid/tests/event-chain.spec.ts
import { test, expect } from '@playwright/test';
// Env: set BASE_URL (served UI), API_URL (FastAPI), and a test user triplet
const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';
const API_URL = process.env.API_URL || 'http://localhost:8001';
async function seedPersona(page, persona) {
    // Example selector IDs; adjust to your UI
    await page.selectOption('#personaSelect', { label: persona });
    await page.click('#applyPersona');
}
test('admin → pilot → mobile event propagation (SSE + UI state)', async ({ browser }) => {
    const admin = await browser.newPage();
    const pilot = await browser.newPage();
    const mobile = await browser.newPage();
    // Open the hybrid UI for each role
    await admin.goto(`${BASE_URL}/?role=admin`);
    await pilot.goto(`${BASE_URL}/?role=pilot`);
    await mobile.goto(`${BASE_URL}/?role=mobile`);
    // Listen for SSE status banners (simple text container)
    const watchBanner = (p) => p.waitForSelector('#statusBanner:has-text("OK")', { timeout: 10000 });
    await Promise.all([watchBanner(admin), watchBanner(pilot), watchBanner(mobile)]);
    // Admin triggers mesh update via API (backend → SSE → UI)
    const res = await admin.request.post(`${API_URL}/api/v2/mesh/update`, {
        data: { user_id: "alice", change: { action: "add_concept", data: { tag: "QEC-Penrose" } } }
    });
    expect(res.ok()).toBeTruthy();
    // All roles should reflect updated mesh summary in UI within ~2s
    await expect(pilot.locator('#meshSummary')).toContainText('QEC-Penrose', { timeout: 5000 });
    await expect(mobile.locator('#meshSummary')).toContainText('QEC-Penrose', { timeout: 5000 });
    // Pilot swaps adapter (UI → backend → logs/SSE → UI refresh)
    await seedPersona(pilot, 'Scientist');
    await pilot.click('#swapAdapter'); // your button
    await expect(admin.locator('#adapterState')).toContainText('active', { timeout: 8000 });
    // Mobile sends prompt → all views should receive SSE "inference started / complete"
    await mobile.fill('#promptInput', 'Explain frame-dragging in simple terms');
    await mobile.click('#sendPrompt');
    await expect(admin.locator('#eventStream')).toContainText('inference_started', { timeout: 5000 });
    await expect(pilot.locator('#eventStream')).toContainText('inference_complete', { timeout: 10000 });
});
