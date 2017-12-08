#!/bin/bash
# bash  hw4_test.sh <testing data> <prediction file>
[ -d input_data ] || mkdir input_data
cp $1 input_data/testing_data.txt
python3 ta_hw4.py simpleRNN-semi test --load_model simpleRNN-semi --result_path $2
