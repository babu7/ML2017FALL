#!/bin/bash
# bash hw4_train.sh <training label data> <training unlabel data>
[ -d input_data ] || mkdir input_data
cp $1 input_data/training_label.txt
cp $2 input_data/training_nolabel.txt
python3 ta_hw4.py check train
