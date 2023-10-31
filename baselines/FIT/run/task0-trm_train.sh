#!/bin/bash
lang=$1
arch=${2:-tagtransformer}

lr=0.001
scheduler=warmupinvsqr
max_steps=2000
warmup=100
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=50
bs=400 # 256

# transformer
layers=4
hs=1024
embed_dim=256
nb_heads=4
dropout=${3:-0.3}

ckpt_dir=checkpoints/sig21

case "$lang" in
"sjo" | "turk" | "vro") trn_path=data/part1/surprise_languages ;;
*) trn_path=data/part1/development_languages ;;
esac
tst_path=data/part1/development_languages

for seed in 40 41 42 43 44 45; do
    CUDA_VISIBLE_DEVICES=3 python ../src/train.py \
        --training \
        --dataset sigmorphon17task1 \
        --train $trn_path/$lang.trn \
        --dev $trn_path/$lang.dev \
        --test $tst_path/$lang.tst \
        --model $ckpt_dir/$arch/$lang \
        --decode greedy --max_decode_len 32 \
        --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
        --label_smooth $label_smooth --total_eval $total_eval \
        --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
        --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
        --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc --seed $seed
done
