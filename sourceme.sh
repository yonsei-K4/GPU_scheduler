#!/bin/bash

# Get the absolute path of the project root (parent of function_app)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set PYTHONPATH to include the sibling directory
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "PYTHONPATH set to: $PYTHONPATH"