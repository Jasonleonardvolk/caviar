# Add this to your .env file or set before running TORI
# This will suppress WARNING level messages from the MCP server

# Windows PowerShell:
$env:TORI_MCP_LOG_LEVEL = "ERROR"

# Or add to .env file:
TORI_MCP_LOG_LEVEL=ERROR

# Or just for the __main__ module:
$env:PYTHONWARNINGS = "ignore::UserWarning:__main__"
