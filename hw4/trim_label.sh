#!/bin/sh
if [ $# -lt 1 ] ; then
	echo Give some file
	exit 1
fi
[ -d input_data ] || mkdir input_data
cp $1 input_data/trimmed_inputs.txt
sed -i 's/^[01] +++$+++ //' input_data/trimmed_inputs.txt
cut -d ' ' -f1 $1 > input_data/trimmed_labels.txt

if [ $# -lt 2 ] ; then
	exit 0
fi
cp $2 input_data/trimmed_test.txt
sed -i '1d' input_data/trimmed_test.txt
sed -i 's/[0-9]\+,//' input_data/trimmed_test.txt
