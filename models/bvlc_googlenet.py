from .py_model import py_model
from .from_caffe.bvlc_googlenet import GoogleNet

class bvlc_googlenet(py_model):
    def __call__(self, hyparams):
        return py_model(hyparams, GoogleNet, 224, name='bvlc_googlenet')