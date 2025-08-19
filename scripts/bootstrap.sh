#!/usr/bin/env bash
set -euo pipefail
pushd "$(dirname "$0")/.." >/dev/null

# 1) Node & Yarn
if ! command -v volta >/dev/null; then
  echo -e "\033[0;36mInstalling Volta for Node & Yarn version management...\033[0m"
  curl https://get.volta.sh | bash
  export PATH="$HOME/.volta/bin:$PATH"
fi
echo -e "\033[0;36mSetting up Node 18.20.3 and Yarn 3.7.1 via Volta...\033[0m"
volta install node@18.20.3 yarn@3.7.1

# 2) Python
if ! command -v pyenv >/dev/null; then
  echo -e "\033[0;33mpyenv not found. Please install pyenv first:\033[0m"
  echo -e "\033[0;33mhttps://github.com/pyenv/pyenv#installation\033[0m"
  echo -e "\033[0;33mThen run this script again.\033[0m"
  exit 1
fi

echo -e "\033[0;36mSetting up Python 3.11.9 via pyenv...\033[0m"
if ! pyenv versions --bare | grep -q 3.11.9; then
  pyenv install 3.11.9
fi
pyenv local 3.11.9
python -m pip install --upgrade pip virtualenv
python -m virtualenv .venv
source .venv/bin/activate

if [ -d "alan_core" ]; then
  echo -e "\033[0;36mInstalling alan_core package...\033[0m"
  pip install -e alan_core/
fi

# 3) JS deps
echo -e "\033[0;36mInstalling JavaScript dependencies...\033[0m"
yarn install --immutable --immutable-cache

echo -e "\n\033[0;32m✅ Environment ready. Run 'yarn dev'.\033[0m"
