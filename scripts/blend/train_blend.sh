#!/usr/bin/env bash
source scripts/data_path.sh

THISNAME="geomvsnet"

LOG_DIR="./checkpoints/blend/"$THISNAME 
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

CUDA_VISIBLE_DEVICES=5 python3 -m torch.distributed.run --nproc_per_node=1 train.py ${@} \
    --which_dataset="blendedmvs" --epochs=16 --logdir=$LOG_DIR \
     --trainpath=$BLENDEDMVS_ROOT --testpath=$BLENDEDMVS_ROOT \
    --trainlist="datasets/lists/blendedmvs/low_res_all.txt" --testlist="datasets/lists/blendedmvs/val.txt" \
    --resume \
    \
    --n_views="7" --batch_size=1 --lr=0.001 --robust_train \
    --lr_scheduler="onecycle"