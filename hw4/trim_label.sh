#!/bin/sh
if [ $# -lt 1 ] ; then
	echo Give some file
	exit 1
fi
[ -d input_data ] || mkdir input_data
cp $1 input_data/trimmed_inputs.txt
sed -i 's/^[01] +++$+++ //' input_data/trimmed_inputs.txt
cut -d ' ' -f1 $1 > input_data/trimmed_labels.txt
