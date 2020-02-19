import time

import numpy as np


class CnnNetInterface(object):
    def __init__(self, layper_param=[]):
        self.__layers_params = layper_param
        self.__layers_params.append(('', 'last_FC'))

    def featuremap_shape(self):
        map_shape = []
        in_map_shape = self.im_height, self.im_width, self.im_dims
        map_shape.append(in_map_shape)
        for layer in self.__layers_params:
            if layer[-1] == 'last_FC':
                break
            elif layer[-1] == 'FC':
                in_map_shape = (1, 1, layer[0])
            elif layer[-1] == 'conv':
                out_depth, filter_size, stride, padding, not_used = layer
                out_height = (in_map_shape[0] - filter_size + 2 * padding) // stride + 1
                out_width = (in_map_shape[1] - filter_size + 2 * padding) // stride + 1
                in_map_shape = (out_height, out_width, out_depth)
                if out_height < filter_size or out_width < filter_size:
                    raise ValueError('the cnn struct is not compatible wieht the image size !')
            elif layer[-1] == 'pool':
                filter_size = 2
                stride = 2
                out_height = (in_map_shape[0] - filter_size) // stride + 1
                out_width = (in_map_shape[1] - filter_size) // stride + 1
                in_map_shape = (out_height, out_width, layer[0])
                if out_height < filter_size or out_width < filter_size:
                    raise ValueError('the cnn struct is not compatible wieht the image size !')
            else:
                raise ValueError('the cnn struct is not compatible wieht the image size !')

            map_shape.append(in_map_shape)
        self.maps_shape = map_shape
        print(map_shape)

    def init_params(self):
        self.__weights = []
        self.__biases = []
        self.__gammas = []
        self.__bates = []
        in_depth = self.im_dims
        out_depth = in_depth

        index = 1
        for layer_param, map_shape in zip(self.__layers_params, self.maps_shape):
            weight = np.array([])
            bias = np.array([])
            gamma = np.array([])
            bate = np.array([])
            if layer_param[-1] == 'last_FC':
                in_depth = out_depth
                out_depth = self.num_class
                (weight, bias) = self.param_init(out_depth, in_depth, map_shape[0] * map_shape[1])
            elif layer_param[-1] == 'FC':
                out_depth = layer_param[0]
                in_depth = map_shape[2]
                weight, bias = self.param_init(out_depth, in_depth, map_shape[0] * map_shape[1])
            elif layer_param[-1] == 'conv':
                filter_size = layer_param[1]
                out_depth = layer_param[0]
                weight, bias = self.param_init(out_depth, in_depth, filter_size * filter_size)
                tmp_map_shape = self.maps_shape[index]
                bn_d = tmp_map_shape[0] * tmp_map_shape[1] * tmp_map_shape[2]
                gamma = np.ones((1, bn_d))
                bate = np.zeros((1, bn_d))
            elif layer_param[-1] == 'pool':
                pass
            else:
                raise ValueError('the cnn struct is not compatible wieht the image size !')

            in_depth = out_depth
            self.__weights.append(weight)
            self.__biases.append(bias)
            self.__gammas.append(gamma)
            self.__bates.append(bate)
            index += 1
        self.__vweights = []
        self.__vbiases = []
        self.__cache_weights = []
        self.__cache_biases = []
        self.__vgammas = []
        self.__vbates = []
        self.__cache_gammas = []
        self.__cache_bates = []
        for weight, bias, gamma, bate in zip(self.__weights, self.__biases, self.__gammas, self.__bates):
            self.__vweights.append(np.zeros_like(weight))
            self.__vbiases.append(np.zeros_like(bias))
            self.__cache_weights.append(np.zeros_like(weight))
            self.__cache_biases.append(np.zeros_like(bias))

            self.__vgammas.append(np.zeros_like(gamma))
            self.__vbates.append(np.zeros_like(bate))
            self.__cache_gammas.append(np.zeros_like(gamma))
            self.__cache_bates.append(np.zeros_like(bate))

    def reg_loss(self, reg, regulation):
        reg_loss = 0
        for weight in self.__weights:
            if weight.size != 0:
                reg_loss += np.sum(regulation.reg(weight, reg))
        return reg_loss

    def forward(self, in_data, labels, activation, reg, regulation, bn=False):
        self.__matric_data = []
        self.__filter_data = []
        self.__matric_data_max_pos = []
        self.__bn_caches = []
        self.__bn_matric_datas = []

        data = in_data
        for i in range(len(self.__layers_params)):
            matric_data = []
            filter_data = []
            max_matric_data_pox = []
            bn_cache = []
            bn_matric_data = []

            if self.__layers_params[i][-1] == 'conv':
                out_data, matric_data, filter_data = self.conv_layer(data, self.__weights[i],
                                                                     self.__biases[i],
                                                                     self.__layers_params[i][0:-1],
                                                                     activation)
                if bn:
                    out_data, bn_cache, bn_matric_data = self.bn_layer(out_data, self.__bates[i], self.__gammas[i])
            elif self.__layers_params[i][-1] == 'pool':
                out_data, max_matric_data_pox = self.pooling_layer(data)
            elif self.__layers_params[i][-1] == 'FC':
                out_data, matric_data, filter_data = self.FC_layer(data, self.__weights[i],
                                                                   self.__biases[i],
                                                                   self.__layers_params[i][0], False,
                                                                   activation)
            elif self.__layers_params[i][-1] == 'last_FC':
                out_data, matric_data, filter_data = self.FC_layer(data, self.__weights[i],
                                                                   self.__biases[i],
                                                                   self.num_class, True,
                                                                   activation)
            else:
                raise KeyError('''struct中 层标识符错误 ''')
            self.__matric_data.append(matric_data)
            self.__filter_data.append(filter_data)
            self.__matric_data_max_pos.append(max_matric_data_pox)
            self.__bn_caches.append(bn_cache)
            self.__bn_matric_datas.append(bn_matric_data)
            data = out_data
        self.__probs = self.softmax_layer(data)
        data_loss = self.data_loss(self.__probs, labels)
        reg_loss = self.reg_loss(reg, regulation)
        return data_loss, reg_loss

    def predict(self, in_data, labels, activation, bn=False):
        data = in_data
        for i in range(len(self.__layers_params)):
            if self.__layers_params[i][-1] == 'conv':
                out_data, matric_data, filter_data = self.conv_layer(data, self.__weights[i],
                                                                     self.__biases[i],
                                                                     self.__layers_params[i][0:-1],
                                                                     activation)
                if bn:
                    out_data, bn_cache, bn_matric_data = self.bn_layer(out_data, self.__bates[i], self.__gammas[i])
            elif self.__layers_params[i][-1] == 'pool':
                out_data, max_matric_data_pox = self.pooling_layer(data)
            elif self.__layers_params[i][-1] == 'FC':
                out_data, matric_data, filter_data = self.FC_layer(data, self.__weights[i],
                                                                   self.__biases[i],
                                                                   self.__layers_params[i][0], False,
                                                                   activation)
            elif self.__layers_params[i][-1] == 'last_FC':
                out_data, matric_data, filter_data = self.FC_layer(data, self.__weights[i],
                                                                   self.__biases[i],
                                                                   self.num_class, True,
                                                                   activation)
            else:
                raise KeyError('''struct中 层标识符错误 ''')
            data = out_data
        predicted_class = np.argmax(data, axis=3)
        accuracy = predicted_class.ravel() == labels
        return np.mean(accuracy)

    def dweight_reg(self, reg, regulation):
        for i in range(len(self.__weights)):
            weight = self.__weights[i]
            if weight.size != 0:
                self.__dweights[-1 - i] += regulation.dreg(weight, reg)

    def backward(self, labels, activation, reg, regulation, bn=True):
        self.__dweights = []
        self.__dbiases = []
        self.__dgammas = []
        self.__dbates = []

        dscores = self.evaluate_dscores(self.__probs, labels)
        dout_data = dscores
        self.__dweights.append(dscores)
        for layer_param, map_shape, matric_data, filter_data, weight, matric_max_data_pox, gamma, bate, bn_matric_data, bn_cache in zip(
                reversed(self.__layers_params),
                reversed(self.maps_shape),
                reversed(self.__matric_data),
                reversed(self.__filter_data),
                reversed(self.__weights),
                reversed(self.__matric_data_max_pos),
                reversed(self.__gammas),
                reversed(self.__bates),
                reversed(self.__bn_matric_datas),
                reversed(self.__bn_caches)):
            dbiase = []
            dweight = []
            dgamma = []
            dbate = []
            if layer_param[-1] == 'last_FC':
                din_data, dweight, dbiase = self.dFC_layer(dout_data, matric_data, filter_data, weight,
                                                           map_shape, True, activation)
            elif layer_param[-1] == 'FC':
                din_data, dweight, dbiase = self.dFC_layer(dout_data, matric_data, filter_data, weight,
                                                           map_shape, False, activation)
            elif layer_param[-1] == 'pool':
                din_data = self.dpooling_layer(dout_data, map_shape, matric_max_data_pox)
            elif layer_param[-1] == 'conv':
                if bn:
                    dgamma, dbate, din_data = self.dbn_layer(dout_data, bn_cache, bn_matric_data, bate, gamma)
                    dout_data = din_data

                din_data, dweight, dbiase = self.dconv_layer(dout_data, matric_data, weight, filter_data,
                                                             layer_param[0:-1], map_shape,
                                                             activation)
            else:
                raise KeyError('''struct中 层标识符错误 ''')
            self.__dweights.append(dweight)
            self.__dbiases.append(dbiase)
            self.__dgammas.append(dgamma)
            self.__dbates.append(dbate)
            dout_data = din_data
        self.dweight_reg(reg, regulation)

    def params_update(self, lr, mu, optimizer, t, update_bn, bn=False):
        self.update_ratio = []

        for i in range(len(self.__weights)):
            weight = self.__weights[i]
            bias = self.__biases[i]
            dweight = self.__dweights[-1 - i]
            dbias = self.__dbiases[-1 - i]
            v_weight = self.__vweights[i]
            v_bias = self.__vbiases[i]
            c_weight = self.__cache_weights[i]
            c_bias = self.__cache_biases[i]
            gamma = self.__gammas[i]
            bate = self.__bates[i]
            dgamma = self.__dgammas[-1 - i]
            dbate = self.__dbates[-1 - i]
            v_gamma = self.__vgammas[i]
            v_bate = self.__vbates[i]
            cache_gamma = self.__cache_gammas[i]
            cache_bate = self.__cache_bates[i]
            if weight.size != 0:
                update_ratio_w = optimizer.update_param(mu, v_weight, weight, dweight, lr, c_weight, t)
                update_ratio_b = optimizer.update_param(mu, v_bias, bias, dbias, lr, c_bias, t)

                self.__weights[i] = weight
                self.__biases[i] = bias
                self.__vweights[i] = v_weight
                self.__vbiases[i] = v_bias
                self.update_ratio.append((update_ratio_w, update_ratio_b))

            if bn and gamma.size != 0:
                update_bn.update_BN(gamma, dgamma, lr)
                update_bn.update_BN(bate, dbate, lr)
                self.__gammas[i] = gamma
                self.__bates[i] = bate
                self.__vgammas[i] = v_gamma
                self.__vbates[i] = v_bate

    def save_checkpoint(self, fname):  # 保存模型所有数据
        with open(fname, 'wb') as f:
            np.save(f, np.array([3, 1, 3, 1, 5, 9, 2, 8, 8]))  # 魔术数
            np.save(f, np.array([self.num_class, self.im_dims, self.im_height, self.im_width]))
            np.save(f, np.array(self.__layers_params))
            np.save(f, np.array(self.maps_shape))
            for array in self.__weights:
                np.save(f, array)
            for array in self.__biases:
                np.save(f, array)
            for array in self.__vweights:
                np.save(f, array)
            for array in self.__vbiases:
                np.save(f, array)

    def load_checkporint(self, fname):
        with open(fname, 'rb') as f:
            magic_number = np.load(f)
            if not all(magic_number == np.array([3, 1, 3, 1, 5, 9, 2, 8, 8])):
                raise ValueError('the file format is wrong! \n')
            im_property = np.load(f, allow_pickle=True)
            self.num_class, self.im_dims, self.im_height, self.im_width = im_property
            self.__layers_params = np.load(f, allow_pickle=True)
            self.maps_shape = np.load(f, allow_pickle=True)
            self.__weights = []
            self.__biases = []
            for i in range(len(self.__layers_params)):
                array = np.load(f, allow_pickle=True)
                self.__weights.append(array)
            for i in range(len(self.__layers_params)):
                array = np.load(f, allow_pickle=True)
                self.__biases.append(array)

            self.__vweights = []
            self.__vbiases = []
            for i in range(len(self.__layers_params)):
                array = np.load(f)
                self.__vweights.append(array)
            for i in range(len(self.__layers_params)):
                array = np.load(f)
                self.__vbiases.append(array)

            print('the struct hyper paramters: \n', self.__layers_params)
