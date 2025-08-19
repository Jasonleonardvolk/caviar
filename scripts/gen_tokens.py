#!/usr/bin/env python3
# Copyright 2025 ALAN Team and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patent Peace / Retaliation Notice:
#   As stated in Section 3 of the Apache 2.0 License, any entity that
#   initiates patent litigation (including a cross-claim or counterclaim)
#   alleging that this software or a contribution embodied within it
#   infringes a patent shall have all patent licenses granted herein
#   terminated as of the date such litigation is filed.

#!/usr/bin/env python3
# Copyright 2025 ALAN Team and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patent Peace / Retaliation Notice:
#   As stated in Section 3 of the Apache 2.0 License, any entity that
#   initiates patent litigation (including a cross-claim or counterclaim)
#   alleging that this software or a contribution embodied within it
#   infringes a patent shall have all patent licenses granted herein
#   terminated as of the date such litigation is filed.

"""
ELFIN Token Generator for Multi-Language Support

PURPOSE:
    Generates stable token definitions for the ELFIN language in multiple target
    languages (Rust, TypeScript) based on the canonical grammar specification.
    Ensures cross-language compatibility with stable token IDs.

WHAT IT DOES:
    1. Parses elfin_v1.ebnf grammar file for token definitions
    2. Categorizes tokens by type (keywords, operators, literals, etc.)
    3. Generates stable numeric IDs using hash-based algorithm
    4. Outputs token enums for Rust and TypeScript
    5. Includes utility functions for token operations

USAGE:
    python gen_tokens.py [--rust] [--ts] [--output-dir DIR]

OUTPUT FILES:
    - Rust: alan_core/elfin_tokens.rs
    - TypeScript: client/src/elfin_tokens.ts

STABILITY:
    Token IDs are stable across versions and must not change to maintain
    compatibility across language boundaries and serialization formats.

AUTHOR: ALAN Team
LAST UPDATED: 2025-01-26

Options:
    --rust           Generate Rust token definitions
    --ts             Generate TypeScript token definitions
    --output-dir DIR Output directory (default: current directory)
"""

import os
import re
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Constants for file paths
GRAMMAR_PATH = Path("elfin/grammar/elfin_v1.ebnf")
RUST_OUTPUT = Path("alan_core/elfin_tokens.rs")
TS_OUTPUT = Path("client/src/elfin_tokens.ts")

# Token type mappings for keywords
TOKEN_CATEGORIES = {
    # Declaration keywords
    "concept": "Keyword",
    "system": "Keyword",
    "relation": "Keyword",
    "lyapunov": "Keyword",
    "koopman": "Keyword",
    "reversible": "Keyword",
    
    # Type keywords
    "float": "Type",
    "int": "Type",
    "bool": "Type",
    "string": "Type",
    "spinvec": "Type",
    
    # Control flow keywords
    "if": "Keyword",
    "else": "Keyword",
    "for": "Keyword",
    "while": "Keyword",
    "return": "Keyword",
    
    # Special symbols
    "ψ-mode": "Symbol",
    "ϕ": "Symbol",
    "ψ_": "Symbol",
    "π": "Symbol",
    
    # Units
    "rad": "Unit",
    "deg": "Unit",
    "s": "Unit",
    "Hz": "Unit",
    
    # Operators default to "Operator" category
}

def extract_tokens_from_grammar(grammar_path: Path) -> Tuple[Dict[str, str], str]:
    """Extract token definitions from the grammar file.
    
    Args:
        grammar_path: Path to the grammar file
        
    Returns:
        Tuple of (tokens_dict, grammar_version):
        - tokens_dict: Dictionary mapping token names to token types
        - grammar_version: Version string extracted from the grammar file
    """
    # Validate grammar file exists
    if not grammar_path.exists():
        print(f"Error: Grammar file not found: {grammar_path}", file=sys.stderr)
        print(f"Please ensure the ELFIN grammar file exists at the expected location.", file=sys.stderr)
        sys.exit(1)
        
    try:
        grammar_text = grammar_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading grammar file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract grammar version
    version = "1.0"  # Default version
    version_match = re.search(r'@version\s+(\d+\.\d+)', grammar_text)
    if version_match:
        version = version_match.group(1)
    
    tokens = {}
    
    # Extract keywords and literals from grammar
    # Look for terminal strings in quotes
    terminal_pattern = r'"([^"]+)"|\'([^\']+)\''
    for match in re.finditer(terminal_pattern, grammar_text):
        term = match.group(1) or match.group(2)
        # Skip single-char operators and punctuation
        if (len(term) == 1 and term in '+-*/=<>(){}[];,:') or term.startswith('/'):
            continue
        
        # Categorize the token
        token_type = "Keyword"  # Default type
        if term in TOKEN_CATEGORIES:
            token_type = TOKEN_CATEGORIES[term]
        elif any(op in term for op in "+-*/=<>!&|^"):
            token_type = "Operator"
        
        token_name = make_token_name(term)
        tokens[token_name] = token_type
    
    # Add special token types from lexical elements section
    tokens["IDENTIFIER"] = "Identifier"
    tokens["STRING_LITERAL"] = "Literal" 
    tokens["INTEGER"] = "Literal"
    tokens["NUMBER"] = "Literal"
    tokens["BOOLEAN"] = "Literal"
    tokens["COMMENT"] = "Comment"
    
    return tokens


def make_token_name(token: str) -> str:
    """Convert a token string to a valid identifier.
    
    Args:
        token: The token string
        
    Returns:
        Valid identifier for the token
    """
    # Special handling for non-ASCII and special characters
    if token == "ψ-mode":
        return "PSI_MODE"
    elif token == "ϕ":
        return "PHI"
    elif token == "ψ_":
        return "PSI_PREFIX"
    elif token == "π":
        return "PI"
    
    # Replace special characters and convert to uppercase
    name = re.sub(r'[^a-zA-Z0-9_]', '_', token).upper()
    
    # Ensure it doesn't start with a digit
    if name and name[0].isdigit():
        name = '_' + name
    
    return name


def generate_token_enum_rust(tokens: Dict[str, str], output_path: Path) -> None:
    """Generate Rust token enumeration.
    
    Args:
        tokens: Dictionary mapping token names to token types
        output_path: Path to write the Rust file
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate stable token IDs based on token names
    token_ids = {}
    for token in sorted(tokens.keys()):
        # Use a hash function to ensure IDs are stable across versions
        # but still change if the token name changes
        token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16) % 10000
        token_ids[token] = token_hash
    
    # Group tokens by category
    token_categories = {}
    for token, category in tokens.items():
        if category not in token_categories:
            token_categories[category] = []
        token_categories[category].append(token)
    
    # Generate Rust code
    rust_code = [
        "// This file is auto-generated. Do not edit manually!",
        "// Generated by scripts/gen_tokens.py from elfin/grammar/elfin_v1.ebnf",
        "",
        "use std::fmt;",
        "",
        "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]",
        "#[repr(u16)]",
        "pub enum TokenKind {",
    ]
    
    # Add token variants by category
    for category in sorted(token_categories.keys()):
        rust_code.append(f"    // {category} tokens")
        for token in sorted(token_categories[category]):
            rust_code.append(f"    {token} = {token_ids[token]},")
        rust_code.append("")
    
    # Close enum definition
    rust_code[-1] = "}"
    
    # Add Display implementation
    rust_code.extend([
        "",
        "impl fmt::Display for TokenKind {",
        "    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {",
        "        match self {",
    ])
    
    # Add match arms for each token
    for token in sorted(tokens.keys()):
        rust_code.append(f"            TokenKind::{token} => write!(f, \"{token}\"),")
    
    # Close implementation
    rust_code.extend([
        "        }",
        "    }",
        "}",
        "",
        "impl TokenKind {",
        "    /// Check if the token is a keyword",
        "    pub fn is_keyword(&self) -> bool {",
        "        matches!(",
        "            self,",
    ])
    
    # Add keyword checks
    keywords = [token for token, category in tokens.items() if category == "Keyword"]
    for keyword in sorted(keywords):
        rust_code.append(f"            TokenKind::{keyword} |")
    
    # Remove trailing pipe from last line and close match
    if keywords:
        rust_code[-1] = rust_code[-1][:-1]
    
    rust_code.extend([
        "        )",
        "    }",
        "",
        "    /// Get the token ID",
        "    pub fn id(&self) -> u16 {",
        "        *self as u16",
        "    }",
        "}",
    ])
    
    # Write to file
    try:
        output_path.write_text("\n".join(rust_code), encoding="utf-8")
        print(f"Generated Rust token definitions at {output_path}")
    except Exception as e:
        print(f"Error writing Rust output: {e}", file=sys.stderr)
        sys.exit(1)


def generate_token_enum_typescript(tokens: Dict[str, str], output_path: Path) -> None:
    """Generate TypeScript token enumeration.
    
    Args:
        tokens: Dictionary mapping token names to token types
        output_path: Path to write the TypeScript file
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate stable token IDs based on token names - use same algorithm as Rust
    token_ids = {}
    for token in sorted(tokens.keys()):
        # Use a hash function to ensure IDs are stable across versions
        token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16) % 10000
        token_ids[token] = token_hash
    
    # Group tokens by category
    token_categories = {}
    for token, category in tokens.items():
        if category not in token_categories:
            token_categories[category] = []
        token_categories[category].append(token)
    
    # Generate TypeScript code
    ts_code = [
        "// This file is auto-generated. Do not edit manually!",
        "// Generated by scripts/gen_tokens.py from elfin/grammar/elfin_v1.ebnf",
        "",
        "/**",
        " * ELFIN Token definitions",
        " * These values must match the Rust definitions",
        " */",
        "export enum TokenKind {",
    ]
    
    # Add token variants by category
    for category in sorted(token_categories.keys()):
        ts_code.append(f"    // {category} tokens")
        for token in sorted(token_categories[category]):
            ts_code.append(f"    {token} = {token_ids[token]},")
        ts_code.append("")
    
    # Close enum definition
    ts_code[-1] = "}"
    
    # Add utility functions
    ts_code.extend([
        "",
        "/**",
        " * Check if a token is a keyword",
        " */",
        "export function isKeyword(token: TokenKind): boolean {",
        "    return [",
    ])
    
    # Add keyword array
    keywords = [token for token, category in tokens.items() if category == "Keyword"]
    for keyword in sorted(keywords):
        ts_code.append(f"        TokenKind.{keyword},")
    
    ts_code.extend([
        "    ].includes(token);",
        "}",
        "",
        "/**",
        " * Get the string representation of a token",
        " */",
        "export function tokenToString(token: TokenKind): string {",
        "    return TokenKind[token];",
        "}",
    ])
    
    # Write to file
    try:
        output_path.write_text("\n".join(ts_code), encoding="utf-8")
        print(f"Generated TypeScript token definitions at {output_path}")
    except Exception as e:
        print(f"Error writing TypeScript output: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate ELFIN token definitions")
    parser.add_argument("--rust", action="store_true", help="Generate Rust token definitions")
    parser.add_argument("--ts", action="store_true", help="Generate TypeScript token definitions")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    # Default to generating both if none specified
    if not args.rust and not args.ts:
        args.rust = True
        args.ts = True
    
    # Extract tokens from grammar
    tokens, grammar_version = extract_tokens_from_grammar(GRAMMAR_PATH)
    print(f"Generating tokens for ELFIN grammar version {grammar_version}")
    
    # Set output paths relative to output directory
    output_dir = Path(args.output_dir)
    rust_output = output_dir / RUST_OUTPUT
    ts_output = output_dir / TS_OUTPUT
    
    # Generate requested outputs
    if args.rust:
        generate_token_enum_rust(tokens, rust_output)
    
    if args.ts:
        generate_token_enum_typescript(tokens, ts_output)


if __name__ == "__main__":
    main()
