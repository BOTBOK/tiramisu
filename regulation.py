import numpy as np


class L2Regulation(object):
    @staticmethod
    def reg(weight, reg):
        return np.sum(weight * weight) * reg / 2

    @staticmethod
    def dreg(weight, reg):
        return reg * weight


class L1Regulation(object):
    @staticmethod
    def reg(weight, reg):
        return np.sum(np.abs(weight)) * reg

    @staticmethod
    def dreg(weight, reg):
        return np.sign(weight) * reg
