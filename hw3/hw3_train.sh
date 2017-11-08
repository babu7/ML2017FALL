#!/bin/bash
# bash  hw3_train.sh <training data>
mkdir input_data
cp $1 input_data/fix.csv
sed -i 's/ /,/g' input_data/fix.csv
python3 to_one_hot.py input_data/fix.csv input_data/train.npz
