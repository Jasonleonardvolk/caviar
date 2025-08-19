#!/usr/bin/env node
// Simple express server to save GPU limits posted from the browser.
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json());

const OUT_DIR = path.join(process.cwd(), "tools", "shaders", "device_limits");
fs.mkdirSync(OUT_DIR, { recursive: true });

app.post("/api/dev/save-gpu-limits", (req, res) => {
  const { name, limits } = req.body || {};
  if (!name || !limits) return res.status(400).json({ ok: false, error: "missing name/limits" });

  const normalized = {
    maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup ?? null,
    maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX ?? null,
    maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY ?? null,
    maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ ?? null,
    maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize ?? limits.maxWorkgroupStorageSize ?? null,
    maxSampledTexturesPerShaderStage: limits.maxSampledTexturesPerShaderStage ?? null,
    maxSamplersPerShaderStage: limits.maxSamplersPerShaderStage ?? null,
    label: name
  };

  const outPath = path.join(OUT_DIR, `${name}.json`);
  fs.writeFileSync(outPath, JSON.stringify(normalized, null, 2));

  // also record a pointer to "latest"
  fs.writeFileSync(path.join(OUT_DIR, "latest.json"), JSON.stringify({ path: `tools/shaders/device_limits/${name}.json` }, null, 2));

  return res.json({ ok: true, path: outPath });
});

const PORT = process.env.LIMITS_PORT || 5178;
app.listen(PORT, () => console.log(`[limits] listening on http://localhost:${PORT}`));
