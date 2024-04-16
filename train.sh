DTU_TRAIN_ROOT="/mnt/nas_8/datasets/dtu_all/mvs_training/dtu_training"
# DTU_TRAIN_ROOT="/home/zhanghanzhi/dtu"
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,5 python train.py --config configs/config_mvsformer.json \
                                         --exp_name MVSFormer \
                                         --data_path ${DTU_TRAIN_ROOT} \
                                         --DDP