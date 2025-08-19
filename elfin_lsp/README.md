# ELFIN Language Server

A Language Server Protocol (LSP) implementation for the ELFIN control system description language.

## Features

- **Real-time diagnostics**: Shows dimensional inconsistencies as warnings
- **Hover Information**: Shows dimensional types when hovering over variables
- **Go-to-Definition**: Navigate to symbol definitions with F12
- **File Watching**: Automatically processes changes to files on disk
- **Integration with VS Code** through the ELFIN Language Support extension

## Installation

```bash
# Install from the repository
git clone <repository-url>
cd elfin-lsp
pip install -e .

# Or directly (once published)
pip install elfin-lsp
```

## Usage

### Command Line

Run the language server directly:

```bash
elfin-lsp run
```

### VS Code Integration

1. Install the ELFIN Language Support extension
2. Open any `.elfin` file
3. The language server will start automatically

## Development

### Prerequisites

- Python 3.8 or later
- pygls 1.0 or later
- ELFIN compiler infrastructure

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd elfin-lsp

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Architecture

The ELFIN Language Server is built on the following components:

- `protocol.py`: LSP protocol definitions
- `server.py`: Core server implementation using pygls
- `cli.py`: Command-line interface

The server integrates with the ELFIN compiler infrastructure to provide language features like:

- Dimensional checking of expressions
- Symbol resolution for hover and go-to-definition
- Future: code actions for common fixes

## License

MIT
