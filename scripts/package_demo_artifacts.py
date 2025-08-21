import sys, os, json, hashlib, zipfile, argparse

ROOT = r'D:\\Dev\\kha'
ART_BASE = os.path.join(ROOT, 'artifacts')

"""
Create a ZIP bundle with a SHA256 manifest for demo artifacts.
Usage:
  python scripts\package_demo_artifacts.py --glob 'artifacts\demo\**\*' --out artifacts\demo\bundle.zip
"""

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default=os.path.join('artifacts', '**', '*'))
    ap.add_argument('--out', default=os.path.join('artifacts', 'demo', 'bundle.zip'))
    args = ap.parse_args()

    # Collect files
    files = []
    for root, _, fnames in os.walk(ART_BASE):
        for fn in fnames:
            full = os.path.join(root, fn)
            # very loose filter: include everything under artifacts/demo + NPZ/JSON in artifacts root
            if '\\demo\\' in full.lower() or fn.lower().endswith(('.npz', '.json', '.jsonl')):
                files.append(full)

    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    zpath = os.path.join(ROOT, args.out)

    manifest = []
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            rel = os.path.relpath(f, ROOT)
            z.write(f, rel)
            manifest.append({'path': rel, 'bytes': os.path.getsize(f), 'sha256': sha256(f)})
        # embed manifest in zip too
        man_json = json.dumps({'files': manifest}, indent=2).encode('utf-8')
        z.writestr('artifacts/demo/manifest.embedded.json', man_json)

    # Save manifest alongside zip
    man_out = os.path.join(os.path.dirname(zpath), 'manifest.json')
    with open(man_out, 'w', encoding='utf-8') as f:
        json.dump({'files': manifest, 'bundle': os.path.relpath(zpath, ROOT)}, f, indent=2)

    print(json.dumps({'bundle': zpath, 'count': len(files), 'manifest': man_out}, indent=2))

if __name__ == '__main__':
    main()
