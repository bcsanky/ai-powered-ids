#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the runtime/reproduction bundle.
# The actual packaging logic lives in ml/src/runtime_bundle/create_runtime_bundle.py.

make runtime-bundle
make runtime-bundle-inspect
make runtime-bundle-report
