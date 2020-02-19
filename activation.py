import numpy as np


class NoneActivation(object):
    @staticmethod
    def forward(param):
        return param

    @staticmethod
    def backward(dparam, param):
        return dparam


class ReluActivation(object):

    @staticmethod
    def forward(param):
        return np.maximum(0, param)

    @staticmethod
    def backward(dparam, param):
        ret_dparam = dparam.copy()
        ret_dparam[param <= 0] = 0
        return ret_dparam


if __name__ == '__main__':
    batch, height, width, depth = (1, 2, 3, 4)
    a = []
    for i in range(batch * height * width * depth):
        a.append(i)
    a = np.array(a).reshape((batch, height, width, depth))
    a[a >= 10] = 1
    print(a)
