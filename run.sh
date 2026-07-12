#!/usr/bin/env bash
set -e

PYTHONWARNINGS="ignore::DeprecationWarning" jupyter nbconvert \
  --to notebook \
  --execute notebooks/08_final_pipeline.ipynb \
  --output final_pipeline_executed.ipynb \
  --output-dir /tmp
