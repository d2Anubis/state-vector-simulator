#!/usr/bin/env bash
# Publish qomputing to PyPI (maintainer only).
# Bump version in pyproject.toml first; then from repo root:  ./scripts/publish-to-pypi.sh
#
# PyPI account and API token: https://pypi.org/manage/account/token/
# Option A: Run this script; when prompted, use username __token__ and password <your token>
# Option B: TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-xxx ./scripts/publish-to-pypi.sh

set -e
cd "$(dirname "$0")/.."

echo "Building (cleaning dist/ first so only current version is uploaded)..."
rm -rf dist/
python3 -m pip install -q build twine
python3 -m build

echo ""
echo "Uploading to PyPI (you may be prompted for username/password)..."
echo "  Username: __token__"
echo "  Password: your PyPI API token (from https://pypi.org/manage/account/token/)"
echo ""
python3 -m twine upload dist/qomputing-*

echo ""
echo "Done. Anyone can now: pip install qomputing"
