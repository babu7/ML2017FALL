#!/bin/bash
# bash  hw3_train.sh <test data> <prediction file>
mkdir input_data
cp $1 input_data/fix.csv
sed -i 's/ /,/g' input_data/fix.csv
python3 test_loader.py input_data/fix.csv input_data/test.npz
./test.py $2
