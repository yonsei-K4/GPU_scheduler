#!/bin/bash

# Get the directory of the script itself
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PYTHONPATH to the script's directory
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "PYTHONPATH set to: $PYTHONPATH"