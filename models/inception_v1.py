import tensorflow as tf

from .abc_models import classification_model
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim

class inception_v1_model(classification_model):
    def __init__(self, hyparams):
        classification_model.__init__(self, hyparams)
        self._name = 'inception_v1_model'
        self._input_size = 224

    def _forward_pass(self, input_layer, trainable=True):
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            input_shape = input_layer.get_shape().as_list()
            batch_size = input_shape[0]
            logits, _ = inception.inception_v1(input_layer, self._hyparams.num_classes, is_training=False, spatial_squeeze=False)
            logits = tf.reshape(logits, (batch_size, self._hyparams.num_classes))

        probabilities = tf.nn.softmax(logits)

        return probabilities