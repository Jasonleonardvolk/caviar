// tools/shaders/lib/limits_schema.js
// Single source of truth for normalizing + validating WebGPU device limits.
// Pure JS (no external deps). Safe to use in resolver, server, and gate.

const PLATFORM_BOUNDS = {
  ios:     { maxX: 1024, maxY: 1024, maxZ: 64,   maxInv: 1024, maxWGStorageKiB: 64  },
  android: { maxX: 1024, maxY: 1024, maxZ: 64,   maxInv: 1024, maxWGStorageKiB: 64  },
  mac:     { maxX: 1024, maxY: 1024, maxZ: 64,   maxInv: 1024, maxWGStorageKiB: 128 },
  windows: { maxX: 1024, maxY: 1024, maxZ: 64,   maxInv: 1536, maxWGStorageKiB: 128 },
  linux:   { maxX: 1024, maxY: 1024, maxZ: 64,   maxInv: 1536, maxWGStorageKiB: 128 },
  default: { maxX: 1024, maxY: 1024, maxZ: 64,   maxInv: 1024, maxWGStorageKiB: 128 },
};

function coerceNumber(v) {
  if (v === null || v === undefined) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function normalizeLimits(raw) {
  const l = raw || {};
  // Normalize naming variants and coerce to numbers
  const maxWGStorage =
    coerceNumber(l.maxComputeWorkgroupStorageSize) ??
    coerceNumber(l.maxWorkgroupStorageSize);

  const normalized = {
    label: l.label ?? '',
    capturedAt: l.capturedAt ?? l.timestamp ?? undefined,
    userAgent: l.userAgent ?? l.ua ?? undefined,
    adapterInfo: l.adapterInfo ?? undefined,
    features: Array.isArray(l.features) ? [...l.features] : undefined,

    maxComputeInvocationsPerWorkgroup: coerceNumber(l.maxComputeInvocationsPerWorkgroup),
    maxComputeWorkgroupSizeX:          coerceNumber(l.maxComputeWorkgroupSizeX),
    maxComputeWorkgroupSizeY:          coerceNumber(l.maxComputeWorkgroupSizeY),
    maxComputeWorkgroupSizeZ:          coerceNumber(l.maxComputeWorkgroupSizeZ),
    maxComputeWorkgroupStorageSize:    maxWGStorage,
    maxSampledTexturesPerShaderStage:  coerceNumber(l.maxSampledTexturesPerShaderStage),
    maxSamplersPerShaderStage:         coerceNumber(l.maxSamplersPerShaderStage),
  };

  return normalized;
}

function validateRequired(n) {
  const missing = [];
  const req = [
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupSizeX',
    'maxComputeWorkgroupSizeY',
    'maxComputeWorkgroupSizeZ',
  ];
  for (const k of req) if (n[k] === undefined) missing.push(k);
  return missing;
}

function validateTypes(n) {
  const wrong = [];
  for (const k of Object.keys(n)) {
    if (k === 'label' || k === 'capturedAt' || k === 'userAgent') continue;
    if (k === 'adapterInfo') continue;
    if (k === 'features') continue;
    const v = n[k];
    if (v === undefined) continue;
    if (typeof v !== 'number') wrong.push(k);
    else if (v < 0) wrong.push(k);
  }
  return wrong;
}

/**
 * Extended sanity checks to catch bad captures.
 * @returns {string[]} array of warnings (not fatal)
 */
function sanityChecks(n, platform = 'default') {
  const w = [];
  const b = PLATFORM_BOUNDS[platform] || PLATFORM_BOUNDS.default;

  if (n.maxComputeWorkgroupSizeX && n.maxComputeWorkgroupSizeX > b.maxX)
    w.push(`maxComputeWorkgroupSizeX unusually high: ${n.maxComputeWorkgroupSizeX} (> ${b.maxX})`);
  if (n.maxComputeWorkgroupSizeY && n.maxComputeWorkgroupSizeY > b.maxY)
    w.push(`maxComputeWorkgroupSizeY unusually high: ${n.maxComputeWorkgroupSizeY} (> ${b.maxY})`);
  if (n.maxComputeWorkgroupSizeZ && n.maxComputeWorkgroupSizeZ > b.maxZ)
    w.push(`maxComputeWorkgroupSizeZ unusually high: ${n.maxComputeWorkgroupSizeZ} (> ${b.maxZ})`);
  if (n.maxComputeInvocationsPerWorkgroup && n.maxComputeInvocationsPerWorkgroup > b.maxInv)
    w.push(`maxComputeInvocationsPerWorkgroup unusually high: ${n.maxComputeInvocationsPerWorkgroup} (> ${b.maxInv})`);

  if (n.maxComputeWorkgroupStorageSize !== undefined) {
    const kib = n.maxComputeWorkgroupStorageSize / 1024;
    if (kib > b.maxWGStorageKiB)
      w.push(`maxComputeWorkgroupStorageSize unusually high: ${kib.toFixed(1)} KiB (> ${b.maxWGStorageKiB} KiB)`);
    if (kib === 0)
      w.push(`maxComputeWorkgroupStorageSize reported as 0 KiB (suspicious)`);
  } else {
    w.push('maxComputeWorkgroupStorageSize missing (some adapters omit it)');
  }

  return w;
}

/**
 * Validate and normalize a limits object.
 * Returns { ok, errors: string[], warnings: string[], normalized }
 */
function validateLimitsSchema(raw, opts = {}) {
  const platform = (opts.platform || 'default').toLowerCase();
  const n = normalizeLimits(raw);

  const errors = [];
  const warnings = [];

  const missing = validateRequired(n);
  if (missing.length) errors.push(`Missing required fields: ${missing.join(', ')}`);

  const wrong = validateTypes(n);
  if (wrong.length) errors.push(`Non-numeric or negative fields: ${wrong.join(', ')}`);

  // Extra logical checks
  const X = n.maxComputeWorkgroupSizeX || 0;
  const Y = n.maxComputeWorkgroupSizeY || 0;
  const Z = n.maxComputeWorkgroupSizeZ || 0;
  const inv = n.maxComputeInvocationsPerWorkgroup || 0;
  if (X === 0 || Y === 0 || Z === 0) {
    warnings.push('One or more maxComputeWorkgroupSize components are 0 (unexpected).');
  }
  if (inv === 0) {
    warnings.push('maxComputeInvocationsPerWorkgroup is 0 (unexpected).');
  }

  warnings.push(...sanityChecks(n, platform));

  return {
    ok: errors.length === 0,
    errors,
    warnings,
    normalized: n,
  };
}

module.exports = {
  normalizeLimits,
  validateLimitsSchema,
  PLATFORM_BOUNDS,
};
