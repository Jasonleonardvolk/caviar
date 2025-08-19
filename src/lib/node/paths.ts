// src/lib/node/paths.ts
// Runtime path resolver for server-side Node.js code
// Resolves ${IRIS_ROOT} tokens to actual filesystem paths
import path from "node:path";

export const IRIS_ROOT = process.env.IRIS_ROOT || process.cwd();
export const resolveFS = (...parts: string[]) => path.join(IRIS_ROOT, ...parts);
export const replaceTokens = (s: string) => s.replace(/\$\{IRIS_ROOT\}/g, IRIS_ROOT);

// Usage examples:
// import { resolveFS, replaceTokens } from "$lib/node/paths";
// const configPath = resolveFS("config", "settings.json");
// const resolvedPath = replaceTokens("${IRIS_ROOT}/data/file.txt");
