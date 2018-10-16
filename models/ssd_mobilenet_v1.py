import os
import tensorflow as tf

from .abc_models import base_model

class ssd_mobilenet_v1_model(base_model):
    def __init__(self, hyparams):
        base_model.__init__(self, hyparams)
        self._hyparams = hyparams
        self._name = 'ssd_mobilenet_v1'
        self._input_size = 300

        self.graph_file_path = os.path.join('.', 'data', 'saved_models', self._name + '.meta')

    def _forward_pass(self, input_layer, trainable=True):
        output_graph_def = tf.GraphDef()

        with open(self.graph_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
            out = tf.import_graph_def(output_graph_def, input_map={"image_tensor": tf.cast(input_layer, tf.uint8)}, return_elements=["detection_boxes"], name="")


    def train(self, train_data):
        raise NotImplementedError("This model is built from pb file, and cannot be trained!")

    def test(self, test_data):
        with tf.name_scope('test'):
            out = self.forward_pass(test_data['images'], trainable=False)
            tf.summary.image('test_img', test_data['images'])

            top_k_op = tf.nn.in_top_k(logits, test_data['labels'], 1)

        return top_k_op