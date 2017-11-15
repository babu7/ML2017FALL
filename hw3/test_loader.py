#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
df = pd.read_csv(sys.argv[1], skiprows=[0], header=None)
df = df.drop(0, axis=1)
inputs = df.values.reshape(-1, 48, 48, 1) / 255
np.savez(sys.argv[2], inputs=inputs)
