# ELFIN Circular Reference Analyzer

This package provides tools for detecting circular references in ELFIN language files, satisfying the Q1 requirement from the ELFIN development roadmap.

## Features

- **Focused Implementation**: Lightweight, zero-friction analyzer that focuses exclusively on circular reference detection
- **Robust Detection**: Identifies both direct (`x = x + 1`) and indirect (`a = b; b = c; c = a`) circular references
- **Language-Aware**: Uses direct string parsing to handle ELFIN syntax without complex parsing overhead
- **Command-line Interface**: Simple `elfin check` command that returns non-zero on detection of issues
- **CI Integration**: Designed to be integrated with CI pipelines for automated checking

## Usage

### Basic Usage

Check a specific ELFIN file:

```bash
python -m alan_backend.elfin.elfin_check check path/to/your/file.elfin
```

Check multiple files using glob patterns:

```bash
python -m alan_backend.elfin.elfin_check check "alan_backend/elfin/templates/**/*.elfin"
```

### Options

- `--output` / `-o`: Directory to save analysis results as JSON files
- `--warnings-as-errors` / `-W`: Treat warnings as errors (exit non-zero)

## Issue Types

The analyzer detects several types of issues:

| Issue Type | Severity | Description |
|------------|----------|-------------|
| `circular_reference` | ERROR | Direct or indirect circular references in variable definitions |
| `undefined_reference` | ERROR/WARNING | References to undefined symbols |
| `missing_dynamics` | ERROR | State variables without corresponding dynamics |
| `duplicate_derivative` | WARNING | Multiple derivative definitions for the same base variable |
| `potential_alias` | WARNING | Variables with identical expressions that could be consolidated |

## Testing

Run the test suite to verify analyzer functionality:

```bash
python -m alan_backend.elfin.analyzer.test_analyzer --run
```

## Advanced Usage

### JSON Output

Export analysis results to JSON for further processing:

```bash
python -m alan_backend.elfin.elfin_check check "*.elfin" -o analysis_results
```

### Integration with CI

In CI pipelines, use the exit code to determine if there are errors:

```bash
python -m alan_backend.elfin.elfin_check check "src/**/*.elfin" -W && echo "All files passed!" || echo "Errors found!"
```

## Implementation Details

The analyzer works by:

1. Parsing ELFIN files into sections (system, helpers, lyapunov, barrier, mode)
2. Extracting symbols and their definitions
3. Analyzing references between symbols
4. Detecting various issues like circular references, aliases, etc.
5. Generating a report of found issues

## Future Enhancements

- LSP integration for real-time feedback in editors
- Auto-fix suggestions for common issues
- Performance optimizations for large codebases
- Extended unit test coverage
