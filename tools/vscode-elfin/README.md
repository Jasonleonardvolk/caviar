# ELFIN Language Support for VS Code

This extension provides language support for the ELFIN control system description language in Visual Studio Code.

## Features

- Syntax highlighting for `.elfin` files
- Real-time diagnostics including dimensional inconsistencies
- Hover information showing dimensional types
- Go-to-definition support
- Automatic monitoring of file changes

## Requirements

- ELFIN Language Server (`elfin-lsp`) must be installed
- Python 3.8 or later

## Installation

### Installing the Language Server

```bash
# Install the language server
pip install -e elfin_lsp/
```

### Installing the Extension

To install the extension locally:

```bash
# Navigate to extension directory
cd tools/vscode-elfin

# Install dependencies
npm install

# Package the extension
npm install -g vsce
vsce package

# Install the extension in VS Code
code --install-extension elfin-language-support-0.1.0.vsix
```

## Configuration

This extension contributes the following settings:

- `elfin.languageServer.enabled`: Enable/disable the ELFIN Language Server
- `elfin.languageServer.path`: Path to the ELFIN Language Server executable

## Security Notes

The extension has several npm dependencies with known vulnerabilities, which are typically inherited from development tooling. These do not affect the extension's runtime behavior in VS Code and will be addressed in future updates.

If you are developing the extension:

```bash
# Address fixable vulnerabilities
npm audit fix

# For comprehensive fixes that may include breaking changes
npm audit fix --force
```

Note that the `--force` option might install breaking changes, so use it cautiously and test thoroughly afterward.

## Contributing

Contributions are welcome! Here are some ways you can contribute:

1. Report bugs
2. Suggest new features
3. Submit pull requests

## License

This extension is licensed under the MIT license.
