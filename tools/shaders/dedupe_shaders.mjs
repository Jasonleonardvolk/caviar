#!/usr/bin/env node
/**
 * dedupe_shaders.mjs
 * Scans the repo for *.wgsl, detects duplicates by basename, resolves to a single canonical path,
 * and emits a manifest + conflict report. Optionally --apply will relocate duplicates into canonical
 * and --backup moves old copies into frontend/shaders.bak/duplicates/<timestamp>/.
 *
 * Usage:
 *   node tools/shaders/dedupe_shaders.mjs --scan                    # Dry run, show duplicates
 *   node tools/shaders/dedupe_shaders.mjs --apply --backup          # Clean up duplicates
 *   node tools/shaders/dedupe_shaders.mjs --report=build/dedupe.json
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Parse CLI arguments
const args = new Map(process.argv.slice(2).map(a => {
  const [k, v] = a.includes('=') ? a.split('=') : [a, true];
  return [k.replace(/^--/, ''), v];
}));

const REPO = path.resolve(process.cwd());
const SCAN_ONLY = args.has('scan');
const APPLY = args.has('apply');
const BACKUP = args.has('backup');
const REPORT = args.get('report') ? path.resolve(String(args.get('report'))) : null;
const ROOTS_CFG = path.join(__dirname, 'shader_roots.json');

const nowTag = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
const BACKUP_DIR = path.join(REPO, 'frontend', 'shaders.bak', 'duplicates', nowTag);

// Color output for terminal
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  gray: '\x1b[90m'
};

function log(color, ...args) {
  console.log(colors[color] + args.join(' ') + colors.reset);
}

function readCfg() {
  if (!fs.existsSync(ROOTS_CFG)) {
    throw new Error(`Missing ${ROOTS_CFG}. Run from repo root.`);
  }
  return JSON.parse(fs.readFileSync(ROOTS_CFG, 'utf-8'));
}

function hashFile(p) {
  try {
    const buf = fs.readFileSync(p);
    return crypto.createHash('sha256').update(buf).digest('hex').slice(0, 16);
  } catch (e) {
    return null;
  }
}

function getFileSize(p) {
  try {
    return fs.statSync(p).size;
  } catch {
    return 0;
  }
}

function walk(dir, exclude) {
  const out = [];
  const stack = [dir];
  
  while (stack.length) {
    const d = stack.pop();
    if (!fs.existsSync(d)) continue;
    
    try {
      const ents = fs.readdirSync(d, { withFileTypes: true });
      for (const e of ents) {
        const p = path.join(d, e.name);
        const rel = path.relative(REPO, p).replace(/\\/g, '/');
        
        // Check exclusions
        if (exclude.some(ex => rel.includes(ex))) continue;
        
        if (e.isDirectory()) {
          stack.push(p);
        } else if (e.isFile() && e.name.endsWith('.wgsl')) {
          out.push(p);
        }
      }
    } catch (e) {
      // Directory not accessible
    }
  }
  return out;
}

function chooseCanonical(entries, cfg) {
  // entries: [{path, hash, size}]
  const canonRoot = cfg.canonical.replace(/\//g, path.sep);
  
  // ALWAYS prefer canonical location first!
  const priority = [
    canonRoot,
    ...cfg.legacy_preferred_order.map(p => p.replace(/\//g, path.sep))
  ];
  
  // Group by hash first
  const byHash = new Map();
  for (const e of entries) {
    if (!e.hash) continue;
    const list = byHash.get(e.hash) || [];
    list.push(e);
    byHash.set(e.hash, list);
  }
  
  // Find the best canonical path
  let canonical = null;
  let allPaths = [];
  
  for (const [hash, list] of byHash.entries()) {
    // Sort by priority
    list.sort((a, b) => {
      const aRel = path.relative(REPO, a.path).replace(/\\/g, '/');
      const bRel = path.relative(REPO, b.path).replace(/\\/g, '/');
      
      const ai = priority.findIndex(pr => aRel.includes(pr));
      const bi = priority.findIndex(pr => bRel.includes(pr));
      
      const aRank = ai >= 0 ? ai : 999;
      const bRank = bi >= 0 ? bi : 999;
      
      if (aRank !== bRank) return aRank - bRank;
      return a.path.length - b.path.length; // Shorter path wins
    });
    
    // First one is the best for this hash
    if (!canonical || list[0].path.includes(canonRoot)) {
      canonical = list[0];
    }
    allPaths = allPaths.concat(list);
  }
  
  // Check for content conflicts
  const conflict = byHash.size > 1;
  
  return { canonical, conflict, variants: allPaths, hashes: byHash };
}

async function main() {
  log('cyan', '\n========================================');
  log('cyan', '   WGSL Shader Deduplication Tool');
  log('cyan', '========================================\n');
  
  const cfg = readCfg();
  log('blue', `Scanning: ${REPO}`);
  log('gray', `Canonical: ${cfg.canonical}`);
  
  // Find all WGSL files
  const files = walk(REPO, cfg.exclude_patterns);
  log('blue', `Found ${files.length} WGSL files\n`);
  
  // Group by basename
  const groups = new Map(); // name -> [{path, hash, size}]
  
  for (const p of files) {
    const base = path.basename(p);
    const hash = hashFile(p);
    const size = getFileSize(p);
    
    const list = groups.get(base) || [];
    list.push({ path: p, hash, size });
    groups.set(base, list);
  }
  
  // Prepare manifest
  const manifest = {
    timestamp: new Date().toISOString(),
    repo: REPO,
    canonical_root: cfg.canonical,
    total_files: files.length,
    unique_names: groups.size,
    duplicates: [],
    conflicts: [],
    actions: {
      moved: [],
      backed_up: [],
      errors: []
    }
  };
  
  // Analyze each group
  for (const [name, entries] of groups.entries()) {
    if (entries.length === 1) continue; // No duplicates
    
    const decision = chooseCanonical(entries, cfg);
    
    if (!decision.canonical) continue;
    
    const canonicalRel = path.relative(REPO, decision.canonical.path).replace(/\\/g, '/');
    
    // Create duplicate entry
    const dupEntry = {
      name,
      canonical: canonicalRel,
      count: entries.length,
      locations: entries.map(e => ({
        path: path.relative(REPO, e.path).replace(/\\/g, '/'),
        hash: e.hash?.slice(0, 8),
        size: e.size
      }))
    };
    
    manifest.duplicates.push(dupEntry);
    
    // Check for conflicts (different content)
    if (decision.conflict) {
      const conflictEntry = {
        name,
        message: `Different content versions found`,
        variants: []
      };
      
      for (const [hash, list] of decision.hashes.entries()) {
        conflictEntry.variants.push({
          hash: hash.slice(0, 8),
          paths: list.map(e => path.relative(REPO, e.path).replace(/\\/g, '/'))
        });
      }
      
      manifest.conflicts.push(conflictEntry);
      log('yellow', `⚠️  CONFLICT: ${name}`);
      for (const variant of conflictEntry.variants) {
        log('gray', `   Hash ${variant.hash}:`);
        for (const p of variant.paths) {
          log('gray', `     - ${p}`);
        }
      }
    } else {
      log('green', `✓ ${name}: ${entries.length} identical copies found`);
      for (const e of entries) {
        const rel = path.relative(REPO, e.path).replace(/\\/g, '/');
        // Check if this path is in the canonical directory
      const isCanonical = e.path.includes(canonRoot);
      const marker = isCanonical ? ' [KEEP]' : ' [REMOVE]';
        log('gray', `   - ${rel}${marker}`);
      }
    }
    
    // Apply changes if requested
    if (APPLY && !SCAN_ONLY) {
      for (const e of entries) {
        if (e.path === decision.canonical.path) continue; // Keep canonical
        
        const rel = path.relative(REPO, e.path).replace(/\\/g, '/');
        
        try {
          // Backup if requested
          if (BACKUP) {
            const backupPath = path.join(BACKUP_DIR, rel);
            fs.mkdirSync(path.dirname(backupPath), { recursive: true });
            fs.copyFileSync(e.path, backupPath);
            manifest.actions.backed_up.push({
              from: rel,
              to: path.relative(REPO, backupPath).replace(/\\/g, '/')
            });
          }
          
          // Only remove if same content
          if (e.hash === decision.canonical.hash) {
            fs.unlinkSync(e.path);
            manifest.actions.moved.push({
              removed: rel,
              canonical: canonicalRel
            });
            log('green', `   Removed: ${rel}`);
          } else {
            log('yellow', `   Skipped (different content): ${rel}`);
          }
        } catch (err) {
          manifest.actions.errors.push({
            file: rel,
            error: err.message
          });
          log('red', `   Error: ${rel} - ${err.message}`);
        }
      }
    }
  }
  
  // Summary
  log('cyan', '\n========================================');
  log('cyan', '   Summary');
  log('cyan', '========================================');
  
  log('blue', `Total files scanned: ${manifest.total_files}`);
  log('blue', `Unique shader names: ${manifest.unique_names}`);
  log('yellow', `Duplicated shaders: ${manifest.duplicates.length}`);
  log('red', `Content conflicts: ${manifest.conflicts.length}`);
  
  if (APPLY && !SCAN_ONLY) {
    log('green', `\nFiles removed: ${manifest.actions.moved.length}`);
    if (BACKUP) {
      log('blue', `Files backed up: ${manifest.actions.backed_up.length}`);
      log('gray', `Backup location: ${path.relative(REPO, BACKUP_DIR)}`);
    }
    if (manifest.actions.errors.length > 0) {
      log('red', `Errors: ${manifest.actions.errors.length}`);
    }
  } else if (SCAN_ONLY) {
    log('gray', '\nThis was a dry run. Use --apply to make changes.');
    log('gray', 'Use --apply --backup to keep backups of removed files.');
  }
  
  // Save report if requested
  if (REPORT) {
    const reportDir = path.dirname(REPORT);
    fs.mkdirSync(reportDir, { recursive: true });
    fs.writeFileSync(REPORT, JSON.stringify(manifest, null, 2));
    log('blue', `\nReport saved to: ${path.relative(REPO, REPORT)}`);
  }
  
  // Exit code
  if (manifest.conflicts.length > 0) {
    process.exit(1); // Conflicts found
  }
  process.exit(0);
}

// Run
main().catch(error => {
  log('red', `\n❌ Fatal error: ${error.message}`);
  console.error(error.stack);
  process.exit(1);
});
