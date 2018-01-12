#!/bin/bash
# bash hw6.sh <image.npy path> <test_case.csv path> <prediction file path>
python3 img_cluster.py tmp test_best --train_data $1 --test_ls $2 --predict $3
