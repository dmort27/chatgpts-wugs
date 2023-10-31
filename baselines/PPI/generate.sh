#!/bin/bash

set -euo pipefail

DATABIN=data-bin
CKPTS=checkpoints

LANGUAGE=$1
TYPE=$2
FRAME=$3

echo $LANGUAGE $TYPE

CHECKPOINT_DIR="${CKPTS}/${LANGUAGE}-models"

mkdir -p "${CKPTS}/${LANGUAGE}-predictions"
PRED="${CKPTS}/${LANGUAGE}-predictions/test"


if [[ "${TYPE}" == "dev" ]]; then
    TYPE=valid
    PRED="${CKPTS}/${LANGUAGE}-predictions/dev"
fi

for MODEL in $(ls "${CHECKPOINT_DIR}"); do
  echo "... generating with model ${MODEL} ..."

  fairseq-generate \
      "${DATABIN}/${LANGUAGE}" \
      --gen-subset "${TYPE}" \
      --source-lang "${LANGUAGE}.input" \
      --target-lang "${LANGUAGE}.output" \
      --path "${CHECKPOINT_DIR}/${MODEL}" \
      --beam 5 \
      > "${PRED}-${MODEL}.txt"

done

# keep only the first 5 best models on dev and the best and last models, delete others
if [[ "${TYPE}" == "valid" ]]; then
    TYPE=dev
    python best_model_on_dev.py $LANGUAGE >> results_$FRAME.txt
fi


