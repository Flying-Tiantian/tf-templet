import tensorflow as tf

from .abc_models import classification_model
from tensorflow.contrib.slim.nets import alexnet

slim = tf.contrib.slim

class inception_v3_model(classification_model):
    def __init__(self, hyparams):
        classification_model.__init__(self, hyparams)
        self._name = 'alexnet_model'
        self._input_size = 227

    def _forward_pass(self, input_layer, trainable=True):
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            input_shape = input_layer.get_shape().as_list()
            batch_size = input_shape[0]
            logits, _ = alexnet.alexnet_v2(input_layer, self._hyparams.num_classes, is_training=False, spatial_squeeze=False)
            logits = tf.reshape(logits, (batch_size, self._hyparams.num_classes))

        probabilities = tf.nn.softmax(logits)

        return probabilities