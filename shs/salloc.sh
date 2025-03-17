#!/bin/bash

# Request interactive session with equivalent resources
salloc \
    --account=bsc88 \
    --qos=acc_debug \
    --partition=acc_debug \
    --time=0-02:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=40 \
    --gres=gpu:2 \
    --job-name=diarization \
    --verbose

# Print date
date +%Y-%m-%d_%H:%M:%S

# Activate virtual environment
source /gpfs/projects/bsc88/speech/environments/diarize_venv/bin/activate

# Set environment variables
export TORCH_HOME="/gpfs/projects/bsc88/speech/speaker_recognition/outputs/diarization/cache/torch"
