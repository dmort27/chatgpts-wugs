#!/bin/bash

set -euo pipefail

DATABIN=data-bin
CKPTS=checkpoints


TYPE=1src
SUBSET='test'
# FRAME=$3


for LANGUAGE in tam deu eng tur; 
do

    echo $LANGUAGE

    ./preprocess_nonce.sh $LANGUAGE $TYPE

    for SEED in 0 1 2 3 4 5 6 7 8 9; 
    do
        CHECKPOINT_DIR="${CKPTS}-${TYPE}/${LANGUAGE}-${SEED}/${LANGUAGE}-models"

        # mkdir -p "${CKPTS}/${LANGUAGE}-predictions"
        PRED="${CKPTS}-${TYPE}/${LANGUAGE}-${SEED}/${LANGUAGE}-predictions/nonce"

        for MODEL in $(ls "${CHECKPOINT_DIR}"); do
        echo "... generating with model ${MODEL} ..."

        fairseq-generate \
            "${DATABIN}/${LANGUAGE}" \
            --gen-subset "${SUBSET}" \
            --source-lang "${LANGUAGE}.input" \
            --target-lang "${LANGUAGE}.output" \
            --path "${CHECKPOINT_DIR}/${MODEL}" \
            --beam 5 \
            > "${PRED}-${MODEL}.txt"

        done

    done

done
