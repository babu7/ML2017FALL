#!/usr/bin/env python3
import sys
import pandas as pd
import logistic as talog

# Load feature and label
X_all = pd.read_csv(sys.argv[1], sep=',', header=0)
X_test = pd.read_csv(sys.argv[2], sep=',', header=0)
# Normalization
X_all, X_test = talog.normalize(X_all, X_test)
talog.infer(X_test, 'logistic_params_8590', sys.argv[3])
