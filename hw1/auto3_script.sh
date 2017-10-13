#!/bin/bash
a1=(hw3av2 hw3bv2 hw3cv2 hw3dv2 hw3ev2 hw3fv2 hw3gv2 hw3hv2)
a2=(1 2 3 4 1 2 3 4)
a3=(0.1 0.01 0.001 0.0001 0.1 0.01 0.001 0.0001)
a4=(1 1 1 1 0 0 0 0)
a5=(features features features features pm25 pm25 pm25 pm25)
a6=(f f f f t t t t)
for i in {0..7}; do
./train.py <<< "1

${a2[$i]}

${a4[$i]}
y

1
${a1[$i]}"
pm25=${a6[$i]} ./best.py input_data/test.csv ${a1[$i]}.csv ${a5[$i]}-100-9hr-x1-ld${a3[$i]}-nr0-${a1[$i]}.npy ${a5[$i]}-100-9hr-x1-ld${a3[$i]}-nr0-${a1[$i]}-scaling.npz
done
