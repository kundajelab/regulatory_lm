#!/bin/bash

ENV_NAME=$1 

conda env create -n $ENV_NAME -f environment.yml

# Install mamba-ssm into the env, ignoring ~/.local just for this command
PYTHONNOUSERSITE=1 PIP_USER=0 conda run -n "$ENV_NAME" \
  python -m pip install --no-build-isolation --no-cache-dir mamba-ssm==1.2.2

