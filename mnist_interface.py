import numpy as np
import gzip, struct


class MNISTInterface(object):
    '''
    load_train_data
    self.num_train_samples 训练集样本数量
    self.num_val_samples 严重集样本数量
    self.train_data 训练集数据
    self.train_labels 训练集标签
    self.val_data
    self.val_labels

    self.test_data 测试集
    self.test_labels 测试集

    self.num_class 结果分类数量
    self.im_height 输入数据高度
    self.im_width 输入数据宽度
    self.im_dims 输入数据深度
    '''

    def load_train_data(self, num_ratio):
        imgs, labels = MNISTInterface.get_mnist_train()
        imgs = (imgs - 128) / 128  # 归一化
        self.num_samples = labels.size
        if isinstance(num_ratio, int):
            self.num_train_samples = num_ratio
        else:
            self.num_train_samples = int(self.num_samples * num_ratio)
        self.num_val_samples = self.num_samples - self.num_train_samples
        shuffle_no = list(range(self.num_samples))
        np.random.shuffle(shuffle_no)
        imgs = imgs[shuffle_no]
        labels = labels[shuffle_no]
        self.train_data = imgs[0:self.num_train_samples]
        self.train_labels = labels[0:self.num_train_samples]
        self.val_data = imgs[self.num_train_samples::]
        self.val_labels = labels[self.num_train_samples::]
        self.__set_data_pro()

    def load_test_data(self):
        imgs, labels = MNISTInterface.get_mnist_test()
        imgs = (imgs - 128) / 128
        self.test_data = imgs
        self.test_labels = labels
        self.__set_data_pro()

    @staticmethod
    def __read(img, label):
        mnist_dir = '/Users/shenjiafeng/data/mnist/'
        with gzip.open(mnist_dir + label) as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            label = np.fromstring(flbl.read(), dtype=np.uint8)
        with gzip.open(mnist_dir + img) as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.fromstring(fimg.read(), dtype=np.uint8).reshape((len(label), rows, cols))
        return image, label

    @staticmethod
    def get_mnist_train():
        train_img, train_label = MNISTInterface.__read('train-images.idx3-ubyte.gz', 'train-labels.idx1-ubyte.gz')
        train_img = train_img.reshape((*train_img.shape, 1))
        return train_img, train_label

    @staticmethod
    def get_mnist_test():
        test_img, test_label = MNISTInterface.__read('t10k-images.idx3-ubyte.gz', 't10k-labels.idx1-ubyte.gz')
        test_img = test_img.reshape((*test_img.shape, 1))
        return test_img, test_label

    def __set_data_pro(self, num_class=10, im_height=28, im_width=28, im_dims=1):
        self.num_class = num_class
        self.im_height = im_height
        self.im_width = im_width
        self.im_dims = im_dims
