import os
import tensorflow as tf
import abc

from .abc_models import base_model, classification_model

class py_model(classification_model):
    def __init__(self, hyparams, model_class, input_size, name=None):
        base_model.__init__(self, hyparams)
        self.model_class = model_class
        self._input_size = input_size
        if name:
            self._name = name
        else:
            self._name = model_class.__name__

    def _forward_pass(self, input_layer, trainable=True):
        self.model = self.model_class({'data': input_layer})
        return self.model.get_output()

    def restore(self, sess, checkpoint_dir):
        global_step = classification_model.restore(self, sess, checkpoint_dir)
        if global_step == -1:
            if os.path.isdir(checkpoint_dir):
                data_path = os.path.join(checkpoint_dir, self._name + '.npy')
            else:
                data_path = checkpoint_dir
            try:
                self.model.load(data_path, sess)
            except Exception as e:
                print(str(e))
            else:
                global_step = 0
        
        return global_step


