#!/bin/bash
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "usage: $(basename "$0") <experiment_name> [step]" >&2
    exit 1
fi

SRC="gs://arl-experiments/runs/$1/profile${2:+/step_$2}"
DEST="./profile/$1"

mkdir -p "$DEST"
gcloud storage cp -r "$SRC/*" "$DEST"
echo "tensorboard --logdir $DEST"
