# % matplotlib inline
import time

from activation import ReluActivation
from cifar_10 import Cifar10Interface
from cnn_block import CnnBlockInterface
from cnn_net_interface import CnnNetInterface
from cnn_train_interafce import CnnTrainInterface
from mnist_interface import MNISTInterface
from optimizer import Nesterov, SGD, Adam, UpdateBN
from regulation import L2Regulation


class CnnTest(MNISTInterface, CnnBlockInterface, CnnNetInterface, CnnTrainInterface):
    pass


class CnnTest2(Cifar10Interface, CnnBlockInterface, CnnNetInterface, CnnTrainInterface):
    pass


def t1():
    # layer_param = [(8, 3, 1, 1, 'conv'), (8, 'pool'), (12, 3, 1, 1, 'conv'), (12, 3, 1, 1, 'conv'),
    #              (12, 3, 1, 1, 'conv'), (12, 'pool'), (36, 3, 1, 1, 'conv'), (36, 3, 1, 1, 'conv'),
    #               (36, 3, 1, 1, 'conv'), (36, 'pool'), (64, 'FC')]
    layer_param = [(6, 5, 1, 0, 'conv'),
                   (6, 'pool'),
                   (16, 5, 1, 0, 'conv'),
                   (16, 'pool'),
                   (120, 'FC'),
                   (84, 'FC')]

    regulation = L2Regulation()
    activation = ReluActivation()
    optimizer = Adam()

    cnn = CnnTest(layer_param)
    cnn.load_train_data(0.7)
    # letnet lr不能太大  10 ** -2 左右时由于步长太长收敛不了
    bn_update = UpdateBN()
    cnn.train_random_search(10, 64, 0.9, [-3.0, -5.0], optimizer, activation, [-3, -5], regulation, 10, 1, True)


def t2():
    regulation = L2Regulation()
    activation = ReluActivation()
    optimizer = Nesterov()

    cnn = CnnTest()
    cnn.load_checkporint('train_1_time_(1657686627894027, 1048576)_param')
    cnn.load_train_data(0.7)
    cnn.train(1, 64, 0.9, 10 ** (-2), optimizer, activation, 10 ** (-5), regulation)


def t3():
    layer_param = [(8, 3, 1, 1, 'conv'),
                   (8, 'pool'),
                   (64, 3, 1, 1, 'conv'),
                   (64, 3, 1, 1, 'conv'),
                   (64, 'pool'),
                   (128, 3, 1, 1, 'conv'),
                   (128, 3, 1, 1, 'conv'),
                   (128, 'pool'),
                   (256, 3, 1, 1, 'conv'),
                   (256, 3, 1, 1, 'conv'),
                   (64, 'FC')]

    # layer_param = [(6, 5, 1, 0, 'conv'),
    #               (6, 'pool'),
    #               (16, 5, 1, 0, 'conv'),
    #               (16, 'pool'),
    #               (120, 'FC'),
    #               (84, 'FC')]

    regulation = L2Regulation()
    activation = ReluActivation()
    optimizer = Adam()
    update_bn = UpdateBN()

    cnn = CnnTest2(layer_param)
    cnn.load_train_data(0.7)
    cnn.train_random_search(10, 64, 0.9, [-3.0, -4.0], optimizer, activation, [-3, -5], regulation, 10, 1, update_bn,
                            True)


if __name__ == '__main__':
    t3()
