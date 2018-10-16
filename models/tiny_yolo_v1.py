from .py_model import py_model
from .from_caffe.tiny_yolo_v1 import TinyYolo

class tiny_yolo_v1_model():
    def __new__(self, hyparams):
        return py_model(hyparams, TinyYolo, 448, name='tiny_yolo_v1')