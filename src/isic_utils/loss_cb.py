# https://github.com/jrzaurin/LightGBM-with-Focal-Loss
# https://www.kaggle.com/code/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
# https://github.com/jhwjhw0123/Imbalance-XGBoost/blob/master/imxgboost/focal_loss.py

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

    return loss


@njit
def robust_pow(num_base, num_pow):
    # numpy does not permit negative numbers to fractional power
    # use this to perform the power algorithmic

    return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)


class FocallossObjective(object):
    def __init__(self, gamma) -> None:
        self.gamma = gamma

    @staticmethod
    @njit
    def calc_ders_range_numba(approxes, targets, weights, gamma):
        exponents = np.exp(approxes)
        p = exponents / (1 + exponents)

        der1 = np.zeros_like(p)
        der2 = np.zeros_like(p)

        pos_indices = targets > 0.0
        neg_indices = ~pos_indices

        der1[pos_indices] = (
            -((1 - p[pos_indices]) ** (gamma - 1))
            * (gamma * np.log(p[pos_indices]) * p[pos_indices] + p[pos_indices] - 1)
            / p[pos_indices]
        )
        der2[pos_indices] = (
            gamma
            * ((1 - p[pos_indices]) ** gamma)
            * ((gamma * p[pos_indices] - 1) * np.log(p[pos_indices]) + 2 * (p[pos_indices] - 1))
        )

        der1[neg_indices] = (
            (p[neg_indices] ** (gamma - 1))
            * (gamma * np.log(1 - p[neg_indices]) - p[neg_indices])
            / (1 - p[neg_indices])
        )
        der2[neg_indices] = p[neg_indices] ** (gamma - 2) * (
            (p[neg_indices] * (2 * gamma * (p[neg_indices] - 1) - p[neg_indices])) / (p[neg_indices] - 1) ** 2
            + (gamma - 1) * gamma * np.log(1 - p[neg_indices])
        )

        if weights is not None:
            der1 *= weights
            der2 *= weights

        return list(zip(der1, der2))

    def calc_ders_range(self, approxes, targets, weights):
        return self.calc_ders_range_numba(
            np.array(approxes),
            np.array(targets),
            np.array(weights) if weights is not None else None,
            self.gamma,
        )


# 途中が学習が止まる、不安定？
# class FocallossObjective(object):
#     def __init__(self, gamma):
#         self.gamma = gamma

#     @staticmethod
#     @njit
#     def focal_binary_object(pred, label, gamma):
#         # compute the prediction with sigmoid
#         sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
#         # gradient
#         # complex gradient with different parts
#         g1 = sigmoid_pred * (1 - sigmoid_pred)
#         g2 = label + ((-1) ** label) * sigmoid_pred
#         g3 = sigmoid_pred + label - 1
#         g4 = 1 - label - ((-1) ** label) * sigmoid_pred
#         g5 = label + ((-1) ** label) * sigmoid_pred
#         # combine the gradient
#         grad = gamma * g3 * robust_pow(g2, gamma) * np.log(g4 + 1e-9) + ((-1) ** label) * robust_pow(
#             g5, (gamma + 1)
#         )
#         # combine the gradient parts to get hessian components
#         hess_1 = robust_pow(g2, gamma) + gamma * ((-1) ** label) * g3 * robust_pow(g2, (gamma - 1))
#         hess_2 = ((-1) ** label) * g3 * robust_pow(g2, gamma) / g4
#         # get the final 2nd order derivative
#         hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma + (gamma + 1) * robust_pow(g5, gamma)) * g1

#         return list(zip(grad, hess))

#     def calc_ders_range(self, approxes, targets, weights):
#         assert weights is None
#         return self.focal_binary_object(
#             np.array(approxes),
#             np.array(targets),
#             self.gamma,
#         )


class FocallossMetric(object):
    def __init__(self, gamma) -> None:
        self.gamma = gamma

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]
        proba = sigmoid(approx)

        loss = fl(proba, target, self.gamma)

        if weight is not None:
            weight_sum = weight
            error_sum = (weight * loss).sum()
        else:
            weight_sum = 1 * len(loss)
            error_sum = loss.sum()

        return error_sum, weight_sum
