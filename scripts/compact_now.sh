#!/bin/bash
# Run TORI Compaction Now
# Simple wrapper to run compaction manually

cd "$(dirname "$0")"
python3 compact_all_meshes.py "$@"
