# https://github.com/jrzaurin/LightGBM-with-Focal-Loss
# https://www.kaggle.com/code/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

import numpy as np
from scipy.misc import derivative
from numba import njit


@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@njit
def logsigmoid(x):
    return np.log(1 / (1 + np.exp(-x)))


@njit
def _fl(x, t, g, max_val):
    loss = x - x * t + max_val + np.log(np.exp(-max_val) + np.exp(-x - max_val))
    invprobs = logsigmoid(-x * (t * 2.0 - 1.0))
    loss = np.exp(invprobs * g) * loss
    return loss


def fl(x, t, g):
    max_val = (-x).clip(min=0)
    loss = _fl(x, t, g, max_val)
    # alpha_t = a * t + (1 - a) * (1 - t)
    # loss = alpha_t * loss

    return loss


def focal_loss_lgb(y_pred, dtrain, gamma):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    """
    g = gamma
    y_true = dtrain.label

    partial_fl = lambda x: fl(x, y_true, g)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess


def focal_loss_lgb_eval_error(y_pred, dtrain, gamma):
    """
    Adapation of the Focal Loss for lightgbm to be used as evaluation loss

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    """
    g = gamma
    y_true = dtrain.label
    loss = fl(y_pred, y_true, g)
    return "focal_loss", np.mean(loss), False
