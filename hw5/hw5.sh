#!/bin/bash
# bash  hw5.sh <test.csv path> <prediction file path> <movies.csv path> <users.csv path>
./train.py d512-dropless-slow test --test_data $1 --predict $2
