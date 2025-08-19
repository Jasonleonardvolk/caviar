#!/usr/bin/env node
/**
 * Concept mesh (.json) → GLB template for Lens Studio / Effect House.
 * Usage:
 *   pnpm dlx tsx tools/exporters/glb-from-conceptmesh.ts ^
 *     -i .\data\concept_graph.json -o .\exports\templates\concept.glb --layout grid --scale 0.12
 */
import fs from 'node:fs';
import path from 'node:path';
import { Document, NodeIO, Accessor, Primitive, Mesh, Node, Buffer, vec3 } from '@gltf-transform/core';
import { dedup, prune } from '@gltf-transform/functions';
import type { ConceptMesh, ConceptNode } from './types/concept-mesh.d.ts';

type Args = { input: string; output: string; scale: number; layout: 'grid' | 'xyz'; };
function parseArgs(argv: string[]): Args {
  const get = (k: string, d?: string) => {
    const i = argv.findIndex(a => a === k);
    return i >= 0 && argv[i + 1] ? argv[i + 1] : d;
  };
  return {
    input: get('-i') || get('--input') || '.\\data\\concept_graph.json',
    output: get('-o') || get('--output') || '.\\exports\\templates\\concept.glb',
    scale: Number(get('--scale', '0.1')),
    layout: (get('--layout', 'grid') as any) === 'xyz' ? 'xyz' : 'grid'
  };
}

function ensureDir(filePath: string) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

// Simple quad primitive (2 triangles) in XY plane.
function createQuad(doc: Document): Mesh {
  const positions = new Float32Array([
    -0.5, -0.5, 0,
     0.5, -0.5, 0,
     0.5,  0.5, 0,
    -0.5,  0.5, 0
  ]);
  const normals = new Float32Array(Array(4).fill(0).flatMap(()=>[0,0,1]));
  const uvs      = new Float32Array([ 0,0, 1,0, 1,1, 0,1 ]);
  const indices  = new Uint16Array([0,1,2, 0,2,3]);

  const mesh = doc.createMesh('ConceptQuad');
  const prim = doc.createPrimitive();

  const buf = doc.createBuffer();
  const posAcc = doc.createAccessor().setType(Accessor.Type.VEC3).setArray(positions).setBuffer(buf);
  const nrmAcc = doc.createAccessor().setType(Accessor.Type.VEC3).setArray(normals).setBuffer(buf);
  const uvAcc  = doc.createAccessor().setType(Accessor.Type.VEC2).setArray(uvs).setBuffer(buf);
  const idxAcc = doc.createAccessor().setType(Accessor.Type.SCALAR).setArray(indices).setBuffer(buf);

  prim.setAttribute('POSITION', posAcc);
  prim.setAttribute('NORMAL', nrmAcc);
  prim.setAttribute('TEXCOORD_0', uvAcc);
  prim.setIndices(idxAcc);
  mesh.addPrimitive(prim);

  return mesh;
}

function seededGridLayout(nodes: ConceptNode[]): vec3[] {
  // Place nodes on a square grid, deterministic by index.
  const N = nodes.length;
  const side = Math.ceil(Math.sqrt(N));
  const coords: vec3[] = [];
  for (let i=0;i<N;i++) {
    const row = Math.floor(i / side);
    const col = i % side;
    coords.push([col, -row, 0]); // negative Y so it reads top-to-bottom visually
  }
  return coords;
}

(async function main() {
  const args = parseArgs(process.argv.slice(2));
  const raw = fs.readFileSync(args.input, 'utf-8');
  const mesh: ConceptMesh = JSON.parse(raw);

  const doc = new Document();
  const scene = doc.createScene('ConceptTemplate');

  const quad = createQuad(doc);
  const scale = args.scale > 0 ? args.scale : 0.1;

  const positions: vec3[] =
    args.layout === 'xyz'
      ? (mesh.nodes.map(n => [n.pos?.x ?? 0, n.pos?.y ?? 0, n.pos?.z ?? 0] as vec3))
      : seededGridLayout(mesh.nodes);

  mesh.nodes.forEach((n, i) => {
    const node = doc.createNode(n.label ?? n.id);
    node.setMesh(quad);
    const p = positions[i];
    node.setTranslation([p[0], p[1], p[2]]);
    node.setScale([scale, scale, scale]);
    scene.addChild(node);
  });

  await doc.transform(dedup(), prune());

  ensureDir(args.output);
  const io = new NodeIO();
  await io.write(args.output, doc);
  console.log(`✅ Wrote GLB: ${args.output} (${mesh.nodes.length} concepts)`);
})().catch((e) => {
  console.error('Exporter failed:', e);
  process.exit(1);
});