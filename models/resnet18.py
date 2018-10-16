from .py_model import py_model
from .from_caffe.resnet18 import ResNet18

class resnet18_model(py_model):
    def __call__(self, hyparams):
        return py_model(hyparams, ResNet18, 224, name='resnet18')