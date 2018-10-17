import numpy as np
import tensorflow as tf

from .abc_models import classification_model


class yolo_tiny_model(classification_model):
    def __init__(self, hyparams):
        classification_model.__init__(self, hyparams)
        self._name = 'yolo_tiny'
        self._input_size = 448
        self.alpha = 0.1

    def conv_layer(self, idx, inputs, filters, size, stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal(
            [size, size, int(channels), filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size//2
        pad_mat = np.array([[0, 0], [pad_size, pad_size],
                            [pad_size, pad_size], [0, 0]])
        inputs_pad = tf.pad(inputs, pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[
                            1, stride, stride, 1], padding='VALID', name=str(idx)+'_conv')
        conv_biased = tf.add(conv, biases, name=str(idx)+'_conv_biased')
        return tf.maximum(self.alpha*conv_biased, conv_biased, name=str(idx)+'_leaky_relu')

    def pooling_layer(self, idx, inputs, size, stride):
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=str(idx)+'_pool')

    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False):
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1]*input_shape[2]*input_shape[3]
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
        if linear:
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx)+'_fc')
        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        return tf.maximum(self.alpha*ip, ip, name=str(idx)+'_fc')

    def _forward_pass(self, input_layer, trainable=True):
        self.conv_1 = self.conv_layer(1, input_layer, 16, 3, 1)
        self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)
        self.conv_3 = self.conv_layer(3, self.pool_2, 32, 3, 1)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)
        self.conv_5 = self.conv_layer(5, self.pool_4, 64, 3, 1)
        self.pool_6 = self.pooling_layer(6, self.conv_5, 2, 2)
        self.conv_7 = self.conv_layer(7, self.pool_6, 128, 3, 1)
        self.pool_8 = self.pooling_layer(8, self.conv_7, 2, 2)
        self.conv_9 = self.conv_layer(9, self.pool_8, 256, 3, 1)
        self.pool_10 = self.pooling_layer(10, self.conv_9, 2, 2)
        self.conv_11 = self.conv_layer(11, self.pool_10, 512, 3, 1)
        self.pool_12 = self.pooling_layer(12, self.conv_11, 2, 2)
        self.conv_13 = self.conv_layer(13, self.pool_12, 1024, 3, 1)
        self.conv_14 = self.conv_layer(14, self.conv_13, 1024, 3, 1)
        self.conv_15 = self.conv_layer(15, self.conv_14, 1024, 3, 1)
        self.fc_16 = self.fc_layer(
            16, self.conv_15, 256, flat=True, linear=False)
        self.fc_17 = self.fc_layer(
            17, self.fc_16, 4096, flat=False, linear=False)
        # skip dropout_18
        self.fc_19 = self.fc_layer(
            19, self.fc_17, 1470, flat=False, linear=True)

        return self.fc_19
