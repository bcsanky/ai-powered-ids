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
