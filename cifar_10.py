import glob

import numpy as np


class Cifar10Interface(object):
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
        file_path = "/Users/shenjiafeng/data/cifar-10/data_batch*"
        images, labels = self.read_data(file_path)
        images = images / 255  # 归一化

        self.num_samples = labels.size
        shuffle_no = list(range(self.num_samples))
        np.random.shuffle(shuffle_no)
        images = images[shuffle_no]
        labels = labels[shuffle_no]

        totalnum = images.shape[0]
        self.num_train_samples = int(totalnum * num_ratio)
        self.train_data = images[:self.num_train_samples]
        self.train_labels = labels[:self.num_train_samples]

        self.num_val_samples = totalnum - self.num_train_samples
        self.val_data = images[self.num_train_samples:]
        self.val_labels = labels[self.num_train_samples:]

        self.__set_data_pro()

    def load_test_data(self):
        file_path = "/Users/shenjiafeng/data/cifar-10/test_batch*"
        images, labels = self.read_data(file_path)

        images = images / 255  # 归一化
        self.test_data = images
        self.test_labels = labels
        self.__set_data_pro()

    def __set_data_pro(self, num_class=10, im_height=32, im_width=32, im_dims=3):
        self.num_class = num_class
        self.im_height = im_height
        self.im_width = im_width
        self.im_dims = im_dims

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def read_data(self, file_path):
        datas = []
        labels = []
        trfiles = glob.glob(file_path)
        for file in trfiles:
            dt = self.unpickle(file)
            datas += list(dt[b"data"])
            labels += list(dt[b"labels"])

        imgs = np.reshape(datas, [-1, 3, 32, 32])

        img_ret = []

        for i in range(imgs.shape[0]):
            im_data = imgs[i,]
            im_data = np.transpose(im_data, [1, 2, 0])

            img_ret.append(im_data)

        return np.array(img_ret), np.array(labels)
