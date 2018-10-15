from .py_model import py_model
from .from_caffe.bvlc_alexnet import AlexNet

class bvlc_alexnet(py_model):
    def __new__(cls, hyparams):
        return py_model(hyparams, AlexNet, 227, name='bvlc_alexnet')
