import numpy as np


class CnnBlockInterface(object):
    @staticmethod
    def conv_layer(in_data, weights, biases, layer_param, activation):
        '''
        :param in_data: 输入的4d矩阵
        :param weights: 权重矩阵
        :param biases:
        :param layer_param: 卷积核信息 :depth, filter_size, stride, padding)
        :param activation: 激活函数
        :return:
        '''

        batch, in_height, in_width, in_depth = in_data.shape
        out_depth, filter_size, stride, padding = layer_param
        out_height = (in_height + 2 * padding - filter_size) // stride + 1
        out_width = (in_width + 2 * padding - filter_size) // stride + 1

        padding_height = in_height + 2 * padding
        padding_width = in_width + 2 * padding
        if padding == 0:
            padding_data = in_data
        else:
            padding_data = np.zeros((batch, padding_height, padding_width, in_depth))
            padding_data[:, padding: -padding, padding:-padding, :] = in_data

        matric_data = np.zeros((batch * out_height * out_width, filter_size * filter_size * in_depth))

        height_ef = padding_height - filter_size + 1
        width_ef = padding_width - filter_size + 1

        for i_batch in range(batch):
            i_batch_size = i_batch * out_height * out_width
            for i_h, i_height in zip(range(out_height), range(0, height_ef, stride)):
                i_height_size = i_batch_size + i_h * out_width
                for i_w, i_width in zip(range(out_width), range(0, width_ef, stride)):
                    i_width_size = i_height_size + i_w
                    matric_data[i_width_size, :] = padding_data[i_batch, i_height: i_height + filter_size,
                                                   i_width: i_width + filter_size, :].ravel()

        filter_data = np.dot(matric_data, weights) + biases  # np的广播机制

        # 激活函数
        activation_data = activation.forward(filter_data)

        out_data = np.zeros((batch, out_height, out_width, out_depth))

        for i_batch in range(batch):
            i_batch_size = i_batch * out_height * out_width
            for i_height in range(out_height):
                i_height_size = i_batch_size + i_height * out_width
                for i_width in range(out_width):
                    i_width_size = i_height_size + i_width
                    out_data[i_batch, i_height, i_width, :] = activation_data[i_width_size, :]

        return out_data, matric_data, filter_data

    @staticmethod
    def dconv_layer(dout_data, matric_data, weights, filter_data, layer_param, out_maps, activation):
        '''

        :param dout_data:
        :param matric_data:
        :param weights:
        :param filter_data:
        :param layer_param:
        :param out_maps:
        :param activation:
        :return:
        '''
        out_depth, filter_size, stride, padding = layer_param
        in_height, in_width, in_depth = out_maps
        batch, out_height, out_width, out_batch = dout_data.shape

        dout_matric_data = np.zeros((batch * out_height * out_width, out_depth))

        for i_batch in range(batch):
            i_batch_size = i_batch * out_height * out_width
            for i_height in range(out_height):
                i_height_size = i_batch_size + i_height * out_width
                for i_width in range(out_width):
                    i_width_size = i_height_size + i_width
                    dout_matric_data[i_width_size, :] = dout_data[i_batch, i_height, i_width, :]

        # 激活函数反馈 入参为ddata和data
        dactivation_data = activation.backward(dout_matric_data, filter_data)

        dbiases = np.sum(dactivation_data, axis=0, keepdims=True)
        dweight = np.dot(matric_data.T, dactivation_data)
        din_matric_data = np.dot(dactivation_data, weights.T)

        padding_height = in_height + 2 * padding
        padding_width = in_width + 2 * padding
        din_padding_data = np.zeros((batch, padding_height, padding_width, in_depth))

        height_ef = padding_height - filter_size + 1
        width_ef = padding_width - filter_size + 1

        for i_batch in range(batch):
            i_batch_size = i_batch * out_height * out_width
            for i_h, i_height in zip(range(out_height), range(0, height_ef, stride)):
                i_height_size = i_batch_size + i_h * out_width
                for i_w, i_width in zip(range(out_width), range(0, width_ef, stride)):
                    i_width_size = i_height_size + i_w
                    din_padding_data[i_batch, i_height: i_height + filter_size, i_width: i_width + filter_size,
                    :] += din_matric_data[i_width_size, :].reshape((filter_size, filter_size, -1))

        if padding != 0:
            din_data = din_padding_data[:, padding: -padding, padding:-padding, :]
        else:
            din_data = din_padding_data

        return din_data, dweight, dbiases

    @staticmethod
    def pooling_layer(in_data, filter_size=2, stride=2):
        '''

        :param in_data:
        :param filter_size:
        :param stride:
        :return:
        '''

        batch, in_height, in_width, in_depth = in_data.shape

        out_depth = in_depth
        out_height = (in_height - filter_size) // stride + 1
        out_width = (in_width - filter_size) // stride + 1

        height_ef = in_height - filter_size + 1
        width_ef = in_width - filter_size + 1

        matric_data = np.zeros((batch * out_height * out_width * in_depth, filter_size * filter_size))

        for i_batch in range(batch):
            i_batch_size = i_batch * out_height * out_width * in_depth
            for i_h, i_height in zip(range(out_height), range(0, height_ef, stride)):
                i_height_size = i_batch_size + i_h * out_width * in_depth
                for i_w, i_width in zip(range(out_width), range(0, width_ef, stride)):
                    i_width_size = i_height_size + i_w * in_depth
                    for i_depth in range(in_depth):
                        i_depth_size = i_width_size + i_depth
                        matric_data[i_depth_size, :] = in_data[i_batch, i_height: i_height + filter_size,
                                                       i_width: i_width + filter_size, i_depth].ravel()

        max_matric_data = np.max(matric_data, axis=1, keepdims=True)
        max_matric_data_pox = matric_data == max_matric_data

        out_data = np.zeros((batch, out_height, out_width, out_depth))

        for i_batch in range(batch):
            i_batch_size = i_batch * out_height * out_width * in_depth
            for i_height in range(out_height):
                i_height_size = i_batch_size + i_height * out_width * in_depth
                for i_width in range(out_width):
                    i_width_size = i_height_size + i_width * in_depth
                    out_data[i_batch, i_height, i_width, :] = max_matric_data[i_width_size: i_width_size + in_depth, 0]
        return out_data, max_matric_data_pox

    @staticmethod
    def dpooling_layer(dout_data, map_shape, matric_max_data_pox, filter_size=2, stride=2):
        '''

        :param dout_data:
        :param map_shape:
        :param matric_max_data_pox:
        :param filter_size:
        :param stride:
        :return:
        '''
        batch, out_height, out_width, out_depth = dout_data.shape
        in_height, in_width, in_depth = map_shape

        dout_matric_data = np.zeros((batch * out_height * out_width * in_depth, 1))
        for i_batch in range(batch):
            i_batch_size = i_batch * out_height * out_width * in_depth
            for i_height in range(out_height):
                i_height_size = i_batch_size + i_height * out_width * in_depth
                for i_width in range(out_width):
                    i_width_size = i_height_size + i_width * in_depth
                    dout_matric_data[i_width_size: i_width_size + in_depth, 0] = dout_data[i_batch, i_height, i_width,
                                                                                 :]
        din_matric_data = np.zeros((batch * out_height * out_width * in_depth, filter_size * filter_size))

        din_matric_data += dout_matric_data

        matric_not_max_data_pox = ~matric_max_data_pox

        din_matric_data[matric_not_max_data_pox] = 0

        din_data = np.zeros((batch, in_height, in_width, in_depth))

        height_ef = in_height - filter_size + 1
        width_ef = in_width - filter_size + 1

        for i_batch in range(batch):
            i_batch_size = i_batch * out_height * out_width * in_depth
            for i_h, i_height in zip(range(out_height), range(0, height_ef, stride)):
                i_height_size = i_batch_size + i_h * out_width * in_depth
                for i_w, i_width in zip(range(out_width), range(0, width_ef, stride)):
                    i_width_size = i_height_size + i_w * in_depth
                    for i_d in range(in_depth):
                        i_depth_size = i_width_size + i_d
                        din_data[i_batch, i_height: i_height + filter_size, i_width:i_width + filter_size,
                        i_d] = din_matric_data[i_depth_size, :].reshape(filter_size, filter_size)

        return din_data

    @staticmethod
    def FC_layer(in_data, weights, biases, out_depth, last, activation):
        '''

        :param in_data:
        :param weights:
        :param biases:
        :param out_depth:
        :param last: True or False
        :param activation:
        :return:
        '''

        batch, in_height, in_width, in_depth = in_data.shape

        matric_data = np.zeros((batch, in_height * in_width * in_depth))
        for i_batch in range(batch):
            matric_data[i_batch, :] = in_data[i_batch, :, :, :].ravel()

        filter_data = np.dot(matric_data, weights) + biases

        if last:
            activation_data = filter_data.copy()
        else:
            activation_data = activation.forward(filter_data)

        out_data = np.zeros((batch, 1, 1, out_depth))

        for i_batch in range(batch):
            out_data[i_batch, 0, 0, :] = activation_data[i_batch, :]

        return out_data, matric_data, filter_data

    @staticmethod
    def dFC_layer(dout_data, matric_data, filter_data, weights, map_shape, last, activation):
        '''

        :param dout_data:
        :param matric_data:
        :param filter_data:
        :param weights:
        :param biases:
        :param map_shape:
        :param last:
        :param activation:
        :return:
        '''
        batch, out_height, out_width, out_depth = dout_data.shape
        in_height, in_width, in_depth = map_shape

        dout_matric_data = np.zeros((batch, out_height * out_width * out_depth))
        for i_batch in range(batch):
            dout_matric_data[i_batch, :] = dout_data[i_batch, :, :, :].ravel()

        if last:
            dactivation_matric_out_data = dout_matric_data
        else:
            dactivation_matric_out_data = activation.backward(dout_matric_data, filter_data)

        dbiase = np.sum(dactivation_matric_out_data, axis=0, keepdims=True)
        dweights = np.dot(matric_data.T, dactivation_matric_out_data)
        dmatric_in_data = np.dot(dactivation_matric_out_data, weights.T)

        din_data = np.zeros((batch, in_height, in_width, in_depth))
        for i_batch in range(batch):
            din_data[i_batch, :, :, :] = dmatric_in_data[i_batch, :].reshape((in_height, in_width, -1))

        return din_data, dweights, dbiase

    @staticmethod
    def bn_layer(in_data, bate, gamma):
        eps = 10 ** (-8)
        # bate 和 gamma都为向量   长度为 height * depth * depth
        batch, height, width, depth = in_data.shape
        matric_data = np.zeros((batch, height * width * depth))
        for i_batch in range(batch):
            matric_data[i_batch] = in_data[i_batch, :, :, :].ravel()

        mu = np.mean(matric_data, axis=0)
        var = np.var(matric_data, axis=0)
        std = np.sqrt(var + eps)
        nat = (matric_data - mu) / std  # 广播机制
        out_matric_data = gamma * nat + bate

        out_data = np.zeros_like(in_data)
        for i_batch in range(batch):
            out_data[i_batch, :, :, :] = out_matric_data[i_batch, :].reshape(height, width, depth)
        bn_cache = (mu, var, std, nat)
        return out_data, bn_cache, matric_data

    @staticmethod
    def dbn_layer(dout_data, bn_cache, matric_data, bate, gamma):
        mu, var, std, nat = bn_cache
        esp = 10 ** (-8)
        batch, height, width, depth = dout_data.shape
        dout_matric_data = np.zeros((batch, height * width * depth))
        for i_batch in range(batch):
            dout_matric_data[i_batch, :] = dout_data[i_batch, :, :, :].ravel()

        dgamma = np.sum(dout_matric_data * nat, axis=0, keepdims=True)
        dbate = np.sum(dout_matric_data, axis=0, keepdims=True)
        dnat = dout_matric_data * gamma

        dmu = -np.sum(dnat, axis=0, keepdims=True) / std
        din_matric_data = dnat / std

        dstd = np.sum(dnat * (mu - matric_data), axis=0) / (std ** 2)

        dvar = dstd * (1 / 2) / ((var ** 0.5) + esp)

        din_matric_data += dvar * 2 * (matric_data - mu) / batch

        din_matric_data += dmu / batch

        din_data = np.zeros_like(dout_data)
        for i_batch in range(batch):
            din_data[i_batch, :, :, :] = din_matric_data[i_batch, :].reshape(height, width, depth)
        return dgamma, dbate, din_data

    @staticmethod
    def softmax_layer(scores):
        '''

        :param score:
        :return:
        '''
        scores -= np.max(scores, axis=3, keepdims=True)
        exp_scores = np.exp(scores) + 10 ** (-8)
        exp_scores_sum = np.sum(exp_scores, axis=3, keepdims=True)
        probs = exp_scores / exp_scores_sum
        return probs

    @staticmethod
    def data_loss(prob, labels):
        '''

        :param prob:
        :param labels:
        :return:
        '''
        probs_correct = prob[range(prob.shape[0]), :, :, labels]

        # -log 预测概率接近1时 损失之接近0  预测概率接近0时损失值无穷大
        logprobs_correct = - np.log(probs_correct)
        # 每次训练不止一批  取批量的均值
        data_loss = np.sum(logprobs_correct) / logprobs_correct.shape[0]
        return data_loss

    @staticmethod
    def evaluate_dscores(prob, labels):
        dscores = prob.copy()
        dscores[range(prob.shape[0]), :, :, labels] -= 1
        # 取批量的均值
        dscores /= dscores.shape[0]
        return dscores

    @staticmethod
    def param_init(out_depth, in_depth, filter_size2):
        '''

        :param out_depth:
        :param in_depth:
        :param filter_size2:
        :return:
        '''
        # todo 为什么需要这么干
        std = np.sqrt(2) / np.sqrt(filter_size2 * in_depth)
        weight = std * np.random.randn(filter_size2 * in_depth, out_depth)
        biases = np.zeros((1, out_depth))
        return weight, biases


if __name__ == '__main__':
    batch, height, width, depth = (1, 2, 3, 4)
    a = []
    for i in range(batch * height * width * depth):
        a.append(i)
    print(np.array(a).shape)
    a = np.array(a).reshape((batch * height, width * depth))

    print(np.mean(a, axis=0))
