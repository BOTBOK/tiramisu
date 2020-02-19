import numpy as np


class SGD(object):
    def update_param(self, mu, vparam, param, dparam, lr, cache, t):
        updata_param = - lr * dparam
        updata_ratio = np.sum(np.abs(updata_param)) / np.sum(np.abs(param))
        param += updata_param
        return updata_ratio


class Nesterov(object):
    def update_param(self, mu, vparam, param, dparam, lr, cache, t):
        pre_v = vparam.copy()
        vparam = mu * vparam - lr * dparam
        updata_param = vparam + mu * (vparam - pre_v)
        updata_ratio = np.sum(np.abs(updata_param)) / np.sum(np.abs(param))
        param += updata_param
        return updata_ratio


class Adam(object):
    decay_rate = 0.999
    eps = 10 ** (-8)

    def update_param(self, mu, vparam, param, dparam, lr, cache, t):
        vparam = vparam * mu + (1 - mu) * dparam
        vparamt = vparam / (1 - mu ** t)

        cache = cache * self.decay_rate + (1 - self.decay_rate) * (dparam ** 2)
        cachet = cache / (1 - self.decay_rate ** t)

        updata_param = - (lr / (np.sqrt(cachet) + self.eps)) * vparamt
        updata_ratio = np.sum(np.abs(updata_param)) / np.sum(np.abs(param))
        param += updata_param

        return updata_ratio


class UpdateBN(object):
    def update_BN(self, param, dparam, lr):
        param += -lr * dparam
