#!/bin/bash

LANG="tur"

BATCH_SIZE=32
LEARNING_RATE=1e-3
EPOCHS=200

SEED=0

CUDA_VISIBLE_DEVICES=0 nohup python3 -u AED.py \
--lang ${LANG} \
--seed ${SEED} \
--lr ${LEARNING_RATE} \
--batch_size ${BATCH_SIZE} \
--early_stopping \
--epochs ${EPOCHS} > ../logs/AED_${LANG}_lr_${LEARNING_RATE}_batch_size_${BATCH_SIZE}_epochs_${EPOCHS}_early_stopping_seed_${SEED}.log &