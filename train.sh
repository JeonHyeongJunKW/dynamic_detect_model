#!/bin/bash
rm -rf triplet/06/* #remove image of query, positive, negative

python3.8 train.py --kitti_color_train_seq_names 05 --kitti_color_valid_seq_names 06 --max_epoch=50 --batch_size=8 --stable_epoch=15 --loss1type=mean --loss1distancetype=euclidean
