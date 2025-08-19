# ELFIN Changelog

All notable changes to the ELFIN language and toolchain are documented in this file.

## [Unreleased]

### Added
- **Language Server Protocol (LSP) Integration**
  - Added `lsp` command to launch LSP server
  - Real-time diagnostics for dimensional errors in VS Code
  - Hover information for functions and keywords
  - Code completion for standard library and language constructs
  - Integration with VS Code through problem matchers

- **Partial Evaluator for Expressions**
  - Added symbolic constant folding for numeric expressions
  - Optimized unit expressions through normalization
  - Automatic dimensional propagation through expressions
  - Function evaluation for built-in math functions
  - Memoization for performance optimization

- **Expanded Standard Library**
  - Physical constants with proper units in `std/constants.elfin`
  - Control theory components (PID, LQR, filters) in `std/control.elfin`
  - Coordinate transformations (quaternions, matrices) in `std/transforms.elfin`

## [0.2.0] - 2025-05-14

### Added
- **Warn-Only Dimensional Checker**
  - Unit expression algebra for dimensional analysis
  - Extended Symbol class with dimensional information
  - Added `check-units` command to CLI
  - Diagnostic output with severity levels
  - Unit comparison with normalization for equivalent expressions

- **Standard Library**
  - Created shared helper functions in `std/helpers.elfin`
  - Refactored all templates to import helpers instead of redefining
  - Removed duplicate helper functions from templates
  - VS Code snippets for standard helpers

- **Formatter and Style Enforcer**
  - Implemented `fmt` CLI command
  - 2-space indentation enforcement
  - Line width limit of 80 columns
  - Vertical alignment of equals signs in param blocks
  - Compact unit annotations
  - Added `--check` mode for CI integration

- **Developer Experience**
  - VS Code integration with tasks and settings
  - Pre-commit hook for automated verification
  - CI workflow for automated checking
  - Golden file tests for formatter
  - Problem matcher for diagnostic output in IDE

### Changed
- Updated CLI interface to support new commands
- Improved error reporting in dimension checker
- Enhanced formatter with better parameter alignment

### Fixed
- Unit comparison now handles equivalent expressions correctly
- Fixed formatting issues with import statements

## [0.1.0] - 2025-04-01

### Added
- Initial release of ELFIN language
- Basic syntax support
- Grammar definition
- Parser implementation
- Simple code generation for Rust
