#!/usr/bin/env node
/**
 * Suppress Known False Positive Warnings
 * Filters out warnings that are confirmed safe
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REPORT_PATH = path.join(__dirname, '../../build/shader_report.json');
const SUPPRESSION_RULES = {
  // These use clamp_index_dyn() which IS bounds checking
  DYNAMIC_INDEXING_BOUNDS: {
    suppress: true,
    reason: "Using clamp_index_dyn() helper for bounds checking",
    pattern: /clamp_index_dyn/
  },
  
  // These are vertex attributes, not storage buffers
  VEC3_STORAGE_ALIGNMENT: {
    suppressFiles: ['avatarShader.wgsl'],
    reason: "Vec3 warnings on vertex attributes (@location), not storage"
  },
  
  // Style preferences we don't care about
  PREFER_CONST: {
    suppress: false, // Keep these as reminders
    reason: "Style preference, not an error"
  }
};

function filterWarnings(report) {
  let