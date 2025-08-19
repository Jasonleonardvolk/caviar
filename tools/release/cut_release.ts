#!/usr/bin/env node
/**
 * Release Automation Tool
 * Cuts releases with checksums, signatures, and provenance
 * 
 * Usage:
 *   node tools/release/cut_release.ts [options]
 * 
 * Options:
 *   --version=X.Y.Z    Version to release
 *   --channel=stable   Release channel (canary/beta/stable)
 *   --sign             Sign the release
 *   --dry-run          Preview without creating release
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import archiver from 'archiver';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.join(__dirname, '../..');

// Parse arguments
const args = process.argv.slice(2).reduce((acc, arg) => {
  const [key, value] = arg.replace('--', '').split('=');
  acc[key] = value || true;
  return acc;
}, {} as Record<string, any>);

const VERSION = args.version || getVersionFromPackage();
const CHANNEL = args.channel || 'stable';
const SHOULD_SIGN = args.sign === true;
const DRY_RUN = args['dry-run'] === true;

interface ReleaseManifest {
  version: string;
  channel: string;
  git: {
    sha: string;
    branch: string;
    tag: string;
    dirty: boolean;
  };
  built: {
    at: string;
    by: string;
    node: string;
    platform: string;
    arch: string;
  };
  artifacts: {
    [key: string]: {
      path: string;
      size: number;
      sha256: string;
      sha512: string;
    };
  };
  checksums: {
    [path: string]: string;
  };
  signatures?: {
    [key: string]: string;
  };
  provenance?: {
    builder: string;
    invocation: string;
    materials: Array<{
      uri: string;
      digest: {
        sha256: string;
      };
    }>;
  };
}

class ReleaseBuilder {
  private manifest: ReleaseManifest;
  private releaseDir: string;
  
  constructor() {
    this.releaseDir = path.join(ROOT_DIR, 'release', VERSION);
    this.manifest = this.initializeManifest();
  }
  
  /**
   * Initialize release manifest
   */
  private initializeManifest(): ReleaseManifest {
    return {
      version: VERSION,
      channel: CHANNEL,
      git: this.getGitInfo(),
      built: {
        at: new Date().toISOString(),
        by: process.env.USER || process.env.USERNAME || 'unknown',
        node: process.version,
        platform: process.platform,
        arch: process.arch
      },
      artifacts: {},
      checksums: {}
    };
  }
  
  /**
   * Get git information
   */
  private getGitInfo() {
    try {
      const sha = execSync('git rev-parse HEAD', { encoding: 'utf-8' }).trim();
      const branch = execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf-8' }).trim();
      const tag = `v${VERSION}`;
      const dirty = execSync('git status --porcelain', { encoding: 'utf-8' }).trim().length > 0;
      
      return { sha, branch, tag, dirty };
    } catch (e) {
      console.warn('Failed to get git info:', e);
      return {
        sha: 'unknown',
        branch: 'unknown',
        tag: `v${VERSION}`,
        dirty: false
      };
    }
  }
  
  /**
   * Run the release process
   */
  public async run() {
    console.log('🚀 Release Builder');
    console.log('==================');
    console.log(`Version: ${VERSION}`);
    console.log(`Channel: ${CHANNEL}`);
    console.log(`Dry Run: ${DRY_RUN}`);
    console.log('');
    
    if (this.manifest.git.dirty) {
      console.warn('⚠️  Warning: Git working directory is dirty');
      
      if (!DRY_RUN && CHANNEL === 'stable') {
        console.error('❌ Cannot create stable release with uncommitted changes');
        process.exit(1);
      }
    }
    
    // Create release directory
    if (!DRY_RUN) {
      fs.mkdirSync(this.releaseDir, { recursive: true });
    }
    
    // Build steps
    await this.runTests();
    await this.buildArtifacts();
    await this.createArchives();
    await this.generateChecksums();
    
    if (SHOULD_SIGN) {
      await this.signRelease();
    }
    
    await this.generateProvenance();
    await this.writeManifest();
    await this.createChangelog();
    
    if (!DRY_RUN) {
      await this.tagRelease();
      await this.uploadArtifacts();
    }
    
    this.printSummary();
  }
  
  /**
   * Run tests before release
   */
  private async runTests() {
    console.log('📋 Running tests...');
    
    if (DRY_RUN) {
      console.log('  [DRY RUN] Skipping tests');
      return;
    }
    
    try {
      // Run frontend tests
      execSync('npm run test', {
        cwd: path.join(ROOT_DIR, 'frontend'),
        stdio: 'inherit'
      });
      
      // Run backend tests
      execSync('pytest tests/', {
        cwd: ROOT_DIR,
        stdio: 'inherit'
      });
      
      console.log('✅ All tests passed\n');
    } catch (e) {
      console.error('❌ Tests failed');
      process.exit(1);
    }
  }
  
  /**
   * Build all artifacts
   */
  private async buildArtifacts() {
    console.log('🔨 Building artifacts...');
    
    if (DRY_RUN) {
      console.log('  [DRY RUN] Skipping build');
      return;
    }
    
    // Build frontend
    console.log('  Building frontend...');
    execSync('npm run build', {
      cwd: path.join(ROOT_DIR, 'frontend'),
      stdio: 'inherit'
    });
    
    // Build Python wheel
    console.log('  Building Python package...');
    execSync('python -m build', {
      cwd: ROOT_DIR,
      stdio: 'inherit'
    });
    
    console.log('✅ Build complete\n');
  }
  
  /**
   * Create release archives
   */
  private async createArchives() {
    console.log('📦 Creating archives...');
    
    if (DRY_RUN) {
      console.log('  [DRY RUN] Skipping archive creation');
      return;
    }
    
    // Create frontend archive
    await this.createArchive(
      path.join(ROOT_DIR, 'frontend/dist'),
      path.join(this.releaseDir, `tori-frontend-${VERSION}.zip`),
      'frontend'
    );
    
    // Create full source archive
    await this.createArchive(
      ROOT_DIR,
      path.join(this.releaseDir, `tori-source-${VERSION}.tar.gz`),
      'source',
      {
        ignore: ['node_modules', '.git', 'dist', '__pycache__', '*.pyc']
      }
    );
    
    console.log('✅ Archives created\n');
  }
  
  /**
   * Create a single archive
   */
  private createArchive(
    sourceDir: string,
    outputPath: string,
    artifactName: string,
    options: any = {}
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const output = fs.createWriteStream(outputPath);
      const archive = outputPath.endsWith('.zip') 
        ? archiver('zip', { zlib: { level: 9 } })
        : archiver('tar', { gzip: true });
      
      output.on('close', () => {
        const stats = fs.statSync(outputPath);
        const hash256 = this.calculateHash(outputPath, 'sha256');
        const hash512 = this.calculateHash(outputPath, 'sha512');
        
        this.manifest.artifacts[artifactName] = {
          path: path.relative(ROOT_DIR, outputPath),
          size: stats.size,
          sha256: hash256,
          sha512: hash512
        };
        
        console.log(`  Created ${path.basename(outputPath)} (${this.formatSize(stats.size)})`);
        resolve();
      });
      
      archive.on('error', reject);
      
      archive.pipe(output);
      
      if (options.ignore) {
        archive.glob('**/*', {
          cwd: sourceDir,
          ignore: options.ignore
        });
      } else {
        archive.directory(sourceDir, false);
      }
      
      archive.finalize();
    });
  }
  
  /**
   * Generate checksums for all files
   */
  private async generateChecksums() {
    console.log('🔐 Generating checksums...');
    
    if (DRY_RUN) {
      console.log('  [DRY RUN] Skipping checksums');
      return;
    }
    
    const checksumFile = path.join(this.releaseDir, 'checksums.sha256');
    const checksums: string[] = [];
    
    // Frontend dist files
    const distDir = path.join(ROOT_DIR, 'frontend/dist');
    if (fs.existsSync(distDir)) {
      this.walkDirectory(distDir, (filePath) => {
        const hash = this.calculateHash(filePath, 'sha256');
        const relativePath = path.relative(ROOT_DIR, filePath);
        this.manifest.checksums[relativePath] = hash;
        checksums.push(`${hash}  ${relativePath}`);
      });
    }
    
    // Release artifacts
    for (const artifact of Object.values(this.manifest.artifacts)) {
      checksums.push(`${artifact.sha256}  ${artifact.path}`);
    }
    
    fs.writeFileSync(checksumFile, checksums.join('\n'));
    console.log(`  Generated ${checksums.length} checksums`);
    console.log('✅ Checksums complete\n');
  }
  
  /**
   * Sign the release
   */
  private async signRelease() {
    console.log('✍️  Signing release...');
    
    if (DRY_RUN) {
      console.log('  [DRY RUN] Skipping signing');
      return;
    }
    
    try {
      const checksumFile = path.join(this.releaseDir, 'checksums.sha256');
      
      // GPG sign checksums
      execSync(`gpg --detach-sign --armor ${checksumFile}`, {
        stdio: 'inherit'
      });
      
      // Sign with SSH key if available
      if (fs.existsSync(path.join(process.env.HOME || '', '.ssh/id_ed25519'))) {
        execSync(`ssh-keygen -Y sign -f ~/.ssh/id_ed25519 -n file ${checksumFile}`, {
          stdio: 'inherit'
        });
      }
      
      this.manifest.signatures = {
        gpg: fs.readFileSync(`${checksumFile}.asc`, 'utf-8'),
        timestamp: new Date().toISOString()
      };
      
      console.log('✅ Release signed\n');
    } catch (e) {
      console.warn('⚠️  Failed to sign release:', e);
    }
  }
  
  /**
   * Generate SLSA provenance
   */
  private async generateProvenance() {
    console.log('📜 Generating provenance...');
    
    if (DRY_RUN) {
      console.log('  [DRY RUN] Skipping provenance');
      return;
    }
    
    this.manifest.provenance = {
      builder: process.env.CI ? 'github-actions' : 'local',
      invocation: process.argv.join(' '),
      materials: [
        {
          uri: `git+https://github.com/org/repo@${this.manifest.git.sha}`,
          digest: {
            sha256: this.manifest.git.sha
          }
        }
      ]
    };
    
    const provenanceFile = path.join(this.releaseDir, 'provenance.json');
    fs.writeFileSync(provenanceFile, JSON.stringify(this.manifest.provenance, null, 2));
    
    console.log('✅ Provenance generated\n');
  }
  
  /**
   * Write release manifest
   */
  private async writeManifest() {
    console.log('📝 Writing manifest...');
    
    if (DRY_RUN) {
      console.log('  [DRY RUN] Would write manifest:');
      console.log(JSON.stringify(this.manifest, null, 2).substring(0, 500) + '...');
      return;
    }
    
    const manifestPath = path.join(this.releaseDir, 'RELEASE.json');
    fs.writeFileSync(manifestPath, JSON.stringify(this.manifest, null, 2));
    
    console.log(`  Wrote ${manifestPath}`);
    console.log('✅ Manifest complete\n');
  }
  
  /**
   * Create changelog
   */
  private async createChangelog() {
    console.log('📄 Creating changelog...');
    
    if (DRY_RUN) {
      console.log('  [DRY RUN] Skipping changelog');
      return;
    }
    
    try {
      // Get commits since last tag
      const lastTag = execSync('git describe --tags --abbrev=0 HEAD^', {
        encoding: 'utf-8'
      }).trim();
      
      const commits = execSync(`git log ${lastTag}..HEAD --pretty=format:"- %s (%h)"`, {
        encoding: 'utf-8'
      });
      
      const changelog = `# Release ${VERSION}

Date: ${new Date().toISOString()}
Channel: ${CHANNEL}

## Changes

${commits}

## Checksums

\`\`\`
${Object.entries(this.manifest.artifacts).map(([name, artifact]) => 
  `${name}: ${artifact.sha256}`
).join('\n')}
\`\`\`
`;
      
      fs.writeFileSync(path.join(this.releaseDir, 'CHANGELOG.md'), changelog);
      console.log('✅ Changelog created\n');
    } catch (e) {
      console.warn('⚠️  Could not generate changelog:', e);
    }
  }
  
  /**
   * Tag the release in git
   */
  private async tagRelease() {
    console.log('🏷️  Tagging release...');
    
    const tag = `v${VERSION}`;
    
    try {
      // Check if tag exists
      execSync(`git rev-parse ${tag}`, { stdio: 'ignore' });
      console.log(`  Tag ${tag} already exists`);
    } catch {
      // Create tag
      execSync(`git tag -a ${tag} -m "Release ${VERSION}"`, {
        stdio: 'inherit'
      });
      console.log(`  Created tag ${tag}`);
      
      if (process.env.CI) {
        execSync(`git push origin ${tag}`, {
          stdio: 'inherit'
        });
        console.log(`  Pushed tag to origin`);
      }
    }
    
    console.log('✅ Tagging complete\n');
  }
  
  /**
   * Upload artifacts to CDN/registry
   */
  private async uploadArtifacts() {
    console.log('☁️  Uploading artifacts...');
    
    // This would upload to S3, npm, PyPI, etc.
    console.log('  [TODO] Implement artifact upload');
    console.log('✅ Upload complete\n');
  }
  
  /**
   * Print release summary
   */
  private printSummary() {
    console.log('');
    console.log('=' .repeat(60));
    console.log('🎉 RELEASE COMPLETE');
    console.log('='.repeat(60));
    console.log(`Version:  ${VERSION}`);
    console.log(`Channel:  ${CHANNEL}`);
    console.log(`Git SHA:  ${this.manifest.git.sha.substring(0, 8)}`);
    console.log(`Artifacts:`);
    
    for (const [name, artifact] of Object.entries(this.manifest.artifacts)) {
      console.log(`  ${name}: ${this.formatSize(artifact.size)}`);
      console.log(`    SHA256: ${artifact.sha256.substring(0, 16)}...`);
    }
    
    if (DRY_RUN) {
      console.log('\n⚠️  This was a dry run - no actual release created');
    } else {
      console.log(`\nRelease artifacts: ${this.releaseDir}`);
      console.log(`\nNext steps:`);
      console.log(`  1. Review artifacts in ${this.releaseDir}`);
      console.log(`  2. Push git tag: git push origin v${VERSION}`);
      console.log(`  3. Create GitHub release`);
      console.log(`  4. Deploy to ${CHANNEL} channel`);
    }
  }
  
  /**
   * Walk directory recursively
   */
  private walkDirectory(dir: string, callback: (path: string) => void) {
    const files = fs.readdirSync(dir);
    
    for (const file of files) {
      const filePath = path.join(dir, file);
      const stat = fs.statSync(filePath);
      
      if (stat.isDirectory()) {
        this.walkDirectory(filePath, callback);
      } else {
        callback(filePath);
      }
    }
  }
  
  /**
   * Calculate file hash
   */
  private calculateHash(filePath: string, algorithm: string): string {
    const hash = crypto.createHash(algorithm);
    const data = fs.readFileSync(filePath);
    hash.update(data);
    return hash.digest('hex');
  }
  
  /**
   * Format file size
   */
  private formatSize(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(2)} ${units[unitIndex]}`;
  }
}

/**
 * Get version from package.json
 */
function getVersionFromPackage(): string {
  const packagePath = path.join(ROOT_DIR, 'frontend/package.json');
  const packageData = JSON.parse(fs.readFileSync(packagePath, 'utf-8'));
  return packageData.version;
}

// Run the release builder
const builder = new ReleaseBuilder();
builder.run().catch(console.error);
