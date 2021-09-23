# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

# for MOT17

# EXP_DIR=exps/e2e_motr_r50_joint
# python3 eval.py \
#     --meta_arch motr \
#     --dataset_file e2e_joint \
#     --epoch 200 \
#     --with_box_refine \
#     --lr_drop 100 \
#     --lr 2e-4 \
#     --lr_backbone 2e-5 \
#     --pretrained ${EXP_DIR}/motr_final.pth \
#     --output_dir ${EXP_DIR} \
#     --batch_size 1 \
#     --sample_mode 'random_interval' \
#     --sample_interval 10 \
#     --sampler_steps 50 90 120 \
#     --sampler_lengths 2 3 4 5 \
#     --update_query_pos \
#     --merger_dropout 0 \
#     --dropout 0 \
#     --random_drop 0.1 \
#     --fp_ratio 0.3 \
#     --query_interaction_layer 'QIM' \
#     --extra_track_attn \
#     --data_txt_path_train ./datasets/data_path/joint.train \
#     --data_txt_path_val ./datasets/data_path/mot17.train \
#     --resume ${EXP_DIR}/motr_final.pth \

# for BDD100k

# EXP_DIR=exps/e2e_motr_r50_joint.bdd100k
# python3 eval_bdd100k.py \
#     --meta_arch motr \
#     --dataset_file bdd100k_mot \
#     --epoch 20 \
#     --with_box_refine \
#     --lr_drop 17 \
#     --save_period 1 \
#     --lr 2e-4 \
#     --lr_backbone 2e-5 \
#     --pretrained ${PRETRAIN} \
#     --output_dir ${EXP_DIR} \
#     --batch_size 1 \
#     --sample_mode 'random_interval' \
#     --sample_interval 4 \
#     --sampler_steps 12 \
#     --sampler_lengths 2 3  \
#     --update_query_pos \
#     --merger_dropout 0 \
#     --dropout 0 \
#     --random_drop 0.1 \
#     --fp_ratio 0.3 \
#     --track_embedding_layer 'AttentionMergerV4' \
#     --extra_track_attn \
#     --data_txt_path_train ./datasets/data_path/bdd100k.train \
#     --data_txt_path_val ./datasets/data_path/bdd100k.val \
#     --mot_path /data/Dataset/bdd100k/bdd100k \
#     --img_path /data/Dataset/bdd100k/bdd100k/images/track/val \
#     --resume ${EXP_DIR}/motr_bdd100k_final.pth \
