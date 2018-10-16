from .py_model import py_model
from .from_caffe.bvlc_googlenet import GoogleNet

class bvlc_googlenet_model(py_model):
    def __new__(self, hyparams):
        return py_model(hyparams, GoogleNet, 224, name='bvlc_googlenet')