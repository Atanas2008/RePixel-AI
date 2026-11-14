#!/bin/sh
cd "$(dirname "$0")" || exit 1
if [ ! -x ".venv/bin/python" ]; then
    echo "Virtual environment not found. See README.md for installation."
    read -r _
    exit 1
fi
exec .venv/bin/python app.py
