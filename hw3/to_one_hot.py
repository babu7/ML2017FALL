#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
df = pd.read_csv(sys.argv[1], skiprows=[0], header=None)
one_hot = pd.get_dummies(df[0])
df = df.drop(0, axis=1)
inputs = df.values.reshape(-1, 48, 48, 1) / 255
labels = np.zeros((inputs.shape[0], 7), dtype=np.int8)
labels[:one_hot.shape[0],:one_hot.shape[1]] = one_hot
np.savez(sys.argv[2], inputs=inputs, labels=labels)
