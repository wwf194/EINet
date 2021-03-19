import abc
import sys
sys.path.append('./Optimizers/')

import torch
from utils import get_from_dict, set_instance_variable, search_dict
import utils_model

from LRSchedulers import LinearLR

class Optimizer(abc.ABC): 
    def __init__(self, dict_=None, load=False, options=None):
        #if options is not None:
        #    self.receive_options(options)
        self.dict = dict_
        self.verbose = get_from_dict(self.dict, 'verbose', default=True, write_default=True)
        #set_instance_variable(self, self.dict)
    def bind_model(self, model):
        if self.model is not None:
            if self.options.verbose:
                print('Optimizer: binding new model. warning: this optimizer has already bound a model, and it will be detached.')
        else:
            self.model = model
    def detach_model(self, model):
        if self.model is None:
            if self.options.verbose:
                print('Optimizer: detaching model. this optimizer hasn\'t bound a model.')
        else:
            self.model = None
    @abc.abstractmethod # must be implemented by child class.
    def train(self):
        return
    def save(self, dir_):
        with open(dir_, 'wb') as f:
            net = self.to(torch.device('cpu'))
            torch.save(net.dict, f)
            net = self.to(self.device)