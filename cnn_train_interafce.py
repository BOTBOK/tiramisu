import numpy as np

import matplotlib.pyplot as plt
import time


class CnnTrainInterface(object):
    @staticmethod
    def gen_lr_reg(lr=[0, -6], reg=[-3, -6], num_try=10):
        minlr = min(lr)
        maxlr = max(lr)
        randn = np.random.rand(num_try * 2)
        lr_array = 10 ** (minlr + (maxlr - minlr) * randn[0:num_try])

        minreg = min(reg)
        maxreg = max(reg)
        reg_array = 10 ** (minreg + (maxreg - minreg) * randn[num_try: 2 * num_try])
        lr_regs = zip(lr_array, reg_array)
        return lr_regs

    def __shuffle_data(self):
        shuffle_no = list(range(self.num_train_samples))
        np.random.shuffle(shuffle_no)
        self.train_labels = self.train_labels[shuffle_no]
        self.train_data = self.train_data[shuffle_no]

        shuffle_no = list(range(self.num_val_samples))
        np.random.shuffle(shuffle_no)
        self.val_labels = self.val_labels[shuffle_no]
        self.val_data = self.val_data[shuffle_no]

    def train_random_search(self, epoch_more, batch, mu, lr, optimizer, activation, reg, regulation, num_try,
                            lr_decay=1, update_bn=None, bn=False):
        self.featuremap_shape()
        lr_regs = self.gen_lr_reg(lr, reg, num_try)
        for lr_reg in lr_regs:
            try:
                self.init_params()
                lr, reg = lr_reg
                print("lr:" + str(lr))
                print("reg:" + str(reg))
                self.train(epoch_more, batch, mu, lr, optimizer, activation, reg, regulation, lr_decay, update_bn, bn)
            except KeyboardInterrupt:
                pass

    def save_plt(self, file_name, data):
        plt.xlabel('x-value')
        plt.ylabel('y-label')
        # plt.scatter(x, y, s, c, marker)
        # x: x轴坐标
        # y：y轴坐标
        # s：点的大小/粗细 标量或array_like 默认是 rcParams['lines.markersize'] ** 2
        # c: 点的颜色
        # marker: 标记的样式 默认是 'o'
        plt.legend()

        plt.scatter(np.arange(len(data)), data, s=20, c="#ff1212", marker='o')

        plt.show()

    def train(self, epoch_more, batch, mu, lr, optimizer, activation, reg, regulation, lr_decay,
              update_bn=None, bn=False):
        data_losses = []
        epoch = 0
        while epoch < epoch_more:
            self.__shuffle_data()
            epoch_time_train = epoch * self.num_train_samples + 1
            for i in range(0, self.num_train_samples, batch):
                print(i)
                batch_data = self.train_data[i:i + batch, :]
                batch_label = self.train_labels[i:i + batch]
                data_loss, reg_loss = self.forward(batch_data, batch_label, activation, reg, regulation, bn)
                data_losses.append(data_loss + reg_loss)
                print(data_loss)

                self.backward(batch_label, activation, reg, regulation, bn)

                t = (epoch_time_train + i) / batch
                self.params_update(lr, mu, optimizer, t, update_bn, bn)
                lr *= lr_decay

            accuracy = self.test(batch, activation, bn)
            file_name = 'accuracy_(' + str(int(1 * 10000)) + '‱)_train_' + str(epoch) + '_time_' + time.strftime(
                "%Y_%m_%d", time.localtime(time.time()))
            self.save_plt(file_name + '_data_loss', data_losses)
            # self.save_checkpoint(file_name + '_param')

            epoch += 1

    def test(self, batch, activation, bn=False):
        self.load_test_data()
        accuracys = np.zeros(shape=(self.test_labels.shape[0],))
        for i in range(0, self.test_labels.shape[0], batch):
            batch_data = self.test_data[i: i + batch]
            label = self.test_labels[i:i + batch]
            accuracys[i: i + batch] = self.predict(batch_data, label, activation, bn)

        accuracy = np.mean(accuracys)
        print("test accuracy:" + str(accuracy))
        return accuracy


if __name__ == '__main__':
    array = [1, 2, 3, 4, 5, 6]
    print(np.mean(np.array(array)))
