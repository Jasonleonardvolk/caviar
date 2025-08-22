// src/lib/server/env.ts
// Centralized environment detection for mock mode
import { env as dyn } from '$env/dynamic/private';

// Treat 1/true/yes (any case) as truthy
const truthy = (v: unknown): boolean =>
  typeof v === 'string' && /^(1|true|yes|y)$/i.test(v);

// IRIS_FORCE_MOCKS overrides IRIS_USE_MOCKS (both checked from Kit + Node)
export const isMock = (): boolean => {
  const force = dyn.IRIS_FORCE_MOCKS ?? process.env.IRIS_FORCE_MOCKS;
  const use   = dyn.IRIS_USE_MOCKS   ?? process.env.IRIS_USE_MOCKS;
  return truthy(force) || truthy(use);
};

// Export for debugging
export const getMockStatus = () => ({
  force: dyn.IRIS_FORCE_MOCKS ?? process.env.IRIS_FORCE_MOCKS,
  use: dyn.IRIS_USE_MOCKS ?? process.env.IRIS_USE_MOCKS,
  isMock: isMock()
});
