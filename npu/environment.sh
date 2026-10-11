#!/usr/bin/env bash
PI05_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PI05_ROOT
# Activate your isolated venv before sourcing this file.
export PYTHONPATH="$PI05_ROOT/src:$PI05_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=600
