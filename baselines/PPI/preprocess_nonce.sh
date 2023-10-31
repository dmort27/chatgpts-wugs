#!/bin/bash

LANGUAGE=$1
TYPE=$2

rm -rf data-bin/$LANGUAGE/*
rm -rf data/$LANGUAGE/*

python3 makedata_$TYPE/prepareTrain.py $LANGUAGE
python3 makedata_$TYPE/prepareDev.py $LANGUAGE
python3 makedata_$TYPE/prepareNonce.py $LANGUAGE

fairseq-preprocess \
    --source-lang="${LANGUAGE}.input" \
    --target-lang="${LANGUAGE}.output" \
    --trainpref=train \
    --validpref=dev \
    --testpref=test \
    --tokenizer=space \
    --thresholdsrc=1 \
    --thresholdtgt=1 \
    --destdir="data-bin/${LANGUAGE}/"

#rm *.input *.output

DATADIR="data/${LANGUAGE}"

mkdir -p $DATADIR

mv *".${LANGUAGE}."* $DATADIR
