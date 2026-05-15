# AI-Powered IDS

AI-based cyber threat detection and analysis platform built around Wazuh and a custom ML service.

## Overview

This project builds an IDS/SIEM-oriented environment with the following components:

- **Wazuh Manager**
- **OpenSearch Indexer**
- **Wazuh Dashboard**
- **Custom ML service** based on FastAPI

The goal is to create a hybrid threat detection platform where traditional log-based detection is complemented by machine learning-based analysis.

## Requirements

- Windows 10/11
- WSL2
- Docker Desktop with WSL backend enabled
- Recommended: at least 8 GB RAM

## Recommended local setup

Work from the Linux filesystem inside WSL:

```bash
/home/<user>/ai-powered-ids
```

## Runtime reproduction bundle

The runtime bundle is for private thesis reproduction and re-checking of the measured pipeline. It contains the source code, tests, final configs, selected model artifacts, real-lab inputs, and verified real-lab result folders needed to inspect or rerun the main evaluation steps.

Unpack it and create a local Python environment:

```bash
unzip ai_powered_ids_runtime_bundle.zip -d ai_powered_ids_runtime_bundle
cd ai_powered_ids_runtime_bundle
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Basic checks:

```bash
python -m py_compile $(find ml/src -name "*.py")
python -m pytest ml/tests
make runtime-bundle-inspect
```

Typical re-check commands include `make real-compare`, `make real-plot`,
`make final-real-measurement-thesis-ready`, and
`make final-measurement-quality MEASUREMENT_QUALITY_THRESHOLDS=docs/measurement_quality_thresholds_real_lab_100.yaml`.
The bundle does not promise a full Wazuh, Zeek, VirtualBox, or network-lab reinstall; those parts still depend on the local lab environment.
