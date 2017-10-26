#!/usr/bin/env python3
import sys
import pandas as pd
import logistic as talog
import generative as tagen

# Load feature and label
X_all = pd.read_csv(sys.argv[1], sep=',', header=0)
X_test = pd.read_csv(sys.argv[2], sep=',', header=0)
# Normalization
X_all, X_test = talog.normalize(X_all, X_test)
tagen.infer(X_test, 'generative_params_8455', sys.argv[3])
