import sys, os, json, argparse, hashlib, time
sys.path.insert(0, r"D:\Dev\kha")
LEDGER = r"D:\\Dev\\kha\\data\\worldstate\\ledger.jsonl"
os.makedirs(os.path.dirname(LEDGER), exist_ok=True)

"""
Append a rule to the worldstate ledger with a content hash.
Usage:
  python demos\rules\inject_rule.py --type bias_potential --payload '{"scale":0.02,"prefer":"kretschmann"}'
"""

def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--type', required=True)
    ap.add_argument('--payload', required=True, help='JSON string payload')
    args = ap.parse_args()
    payload = json.loads(args.payload)

    entry = {
        'ts': time.time(),
        'type': args.type,
        'payload': payload,
    }
    h = hashlib.sha256(canonical_json(entry).encode('utf-8')).hexdigest()
    entry['sha256'] = h

    with open(LEDGER, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry, indent=2))

if __name__ == '__main__':
    main()
