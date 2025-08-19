@echo off
echo 🔑 Generating Vault Fingerprint...
echo ================================
echo.

python "%~dp0\vault_inspector.py" --fingerprint %*
