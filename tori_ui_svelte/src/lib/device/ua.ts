// UA parsing for iPhone/iPad hardware model extraction

export function parseHardwareModel(uaText: string): string | null {
  // iPhone: iPhone13,* .. iPhone17,* ; iPad: the enumerated sets
  const m = uaText.match(/\b(iPhone1[3-7],\d+|iPad1[346],\d+|iPad15,\d+|iPad16,\d+)\b/);
  return m ? m[1] : null;
}

export function detectiPhoneModelFromUA(uaString: string): string {
  // Extract hardware model from UA string
  const hw = parseHardwareModel(uaString);
  if (hw) return hw;
  
  // Fallback: try to extract from common UA patterns
  if (uaString.includes('iPhone')) {
    // Try to match iPhone model patterns
    const modelMatch = uaString.match(/iPhone(\d+),(\d+)/);
    if (modelMatch) {
      return `iPhone${modelMatch[1]},${modelMatch[2]}`;
    }
  }
  
  if (uaString.includes('iPad')) {
    // Try to match iPad model patterns
    const modelMatch = uaString.match(/iPad(\d+),(\d+)/);
    if (modelMatch) {
      return `iPad${modelMatch[1]},${modelMatch[2]}`;
    }
  }
  
  // Unknown device
  return 'unknown';
}