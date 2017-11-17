#!/usr/bin/env python3
import numpy as np
from keras.models import load_model

good_model = load_model('weighs_63509.h5')
test = np.load('input_data/test.npz')['inputs']
y = good_model.predict(test)
decode = np.argmax(y, axis=1)

import sys
with open(sys.argv[1], 'w') as f:
    print("id,label", file=f)
    for i in range(decode.shape[0]):
        print("%d,%d" % (i, decode[i]), file=f)
