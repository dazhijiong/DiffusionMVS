##!/usr/bin/env bash

DTU_TESTING="/home/zhanghanzhi/dtu/"

TANK_TESTING='/data1/local_userdata/wangshaoqian/tankandtemples/'

CKPT_FILE="/home/zhanghanzhi/Effi-MVS/checkpoints/Effi-MVS/model_000015.ckpt"

OUT_DIR='Effi-MVS_result'

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
##DTU
python test.py --dataset=general_eval --batch_size=1 --testpath=$DTU_TESTING  --ndepths=48 --CostNum=4 --numdepth=384 --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE --outdir $OUT_DIR --data_type dtu \
              --num_view=5
##tank  train_on_dtu
#python test.py --dataset=tank --batch_size=1 --testpath=$TANK_TESTING  --ndepths=96 --CostNum=4 --numdepth=384 --loadckpt $CKPT_FILE --outdir $OUT_DIR --data_type tank \
#              --num_view=11
##tank  train_on_blend
#python test.py --dataset=tank --batch_size=1 --testpath=$TANK_TESTING  --ndepths=96 --CostNum=4 --numdepth=768 --loadckpt $CKPT_FILE --outdir $OUT_DIR --data_type tank \
#              --num_view=11
