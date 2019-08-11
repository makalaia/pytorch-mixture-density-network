import numpy as np


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
    if y_true.shape != y_pred.shape:
        raise ValueError('The shapes must be the same to perform the calculation Real: %s\tPredict: %s' %
                         (y_true.shape, y_pred.shape))
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
    if y_true.shape != y_pred.shape:
        raise ValueError('The shapes must be the same to perform the calculation Real: %s\tPredict: %s' %
                         (y_true.shape, y_pred.shape))
    return np.sqrt(np.mean((y_true-y_pred)**2))