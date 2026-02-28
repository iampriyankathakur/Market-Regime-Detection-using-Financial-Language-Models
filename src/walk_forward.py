import numpy as np

def walk_forward_split(X, y, train_size=0.7):

    split_idx = int(len(X) * train_size)

    X_train = X[:split_idx]
    y_train = y[:split_idx]

    X_test = X[split_idx:]
    y_test = y[split_idx:]

    return X_train, y_train, X_test, y_test
