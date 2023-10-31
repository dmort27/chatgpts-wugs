model=tagtransformer # input-invariant transformer, or `model=transformer` the vanilla transformer
# No data augmentation
for lang in tam; do
bash ./task0-trm_train.sh $lang $model
done