import torch
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from utils import ensure_path, get_from_dict, search_dict
from utils_model import print_model_param, print_optimizer_params

class Trainer():
    def __init__(self, dict_, load=False, options=None):
        '''
        if options is not None:
            self.receive_options(options)
        else:
            raise Exception('Trainer: options is none.')
        '''
        self.dict = dict_
        
        '''
        self.epoch_now = get_from_dict(self.dict, 'epoch_now', default=self.epoch_start, write_default=True)
        self.epoch_start = get_from_dict(self.dict, 'epoch_start', default=1, write_default=True)
        self.epoch_end = get_from_dict(self.dict, 'epoch_end', default=self.epoch_um, write_default=True)
        '''        
        self.epoch_now = 0
        #print(self.dict.keys())
        self.epoch_num = self.dict['epoch_num']
        self.epoch_end = self.epoch_num - 1

        # save directory setting
        self.save_model_path = search_dict(self.dict, ['save_model_path', 'save_dir_model', 'save_path_model'], 
            default='./SavedModels/', write_default=True, write_default_key='save_model_path')
        #print(self.save_model_path)
        ensure_path(self.save_model_path)

        self.save_model = get_from_dict(self.dict, 'save_model', default=True, write_default=True)
        self.save_after_train = get_from_dict(self.dict, 'save_after_train', default=True, write_default=True)
        self.save_before_train = get_from_dict(self.dict, 'save_before_train', default=True, write_default=True)

        if self.save_model:
            self.save_interval = get_from_dict(self.dict, 'save_model_interval', default=True, write_default=True)

        self.anal_path = search_dict(self.dict, ['anal_path'], default='./', write_default=True)
        #print(self.anal_path)
        ensure_path(self.anal_path)

    def train(self):
        if self.save_before_train:
            self.model.save(save_path=self.save_model_path, save_name=self.model.dict['name'] + '_epoch=beforeTrain')
             
        self.test_performs = self.dict['test_performs'] = {}
        self.train_performs = self.dict['train_performs'] = {}

        print_model_param(self.model)
        print_model_param(self.optimizer.decoder_rec)
        print_model_param(self.optimizer.decoder_out)
        print(self.optimizer.decoder_rec.N_nums)
        print(self.optimizer.decoder_out.N_nums)
        print_optimizer_params(self.optimizer.optimizer_rec)
        print_optimizer_params(self.optimizer.optimizer_out)
        print_optimizer_params(self.optimizer.optimizer)

        while self.epoch_now <= self.epoch_end:
            print('epoch=%d'%(self.epoch_now), end=' ')
            train_data, test_data = self.data_loader.get_data()
            # train model
            self.model.reset_perform()
            batch_num = len(train_data)
            report_interval = int(batch_num / 10)
            for batch_index, data in enumerate(train_data):
                #print('using batch No.%d\n'%(batch_num))
                inputs, labels = data
                self.optimizer.train({
                    'input': inputs.to(self.model.device),
                    'output': labels.to(self.model.device),
                })
                if batch_index % report_interval == 0:
                    self.optimizer.model.anal_weight_change()
                    #self.optimizer.decoder_rec.print_grad()
                    #self.optimizer.decoder_out.print_grad()
                    '''
                    self.optimizer.decoder_rec.anal_weight_change(title='decoder_rec weight change')
                    self.optimizer.decoder_out.anal_weight_change(title='decoder_out weight change')
                    if hasattr(self.optimizer, 'output_grad_example'):
                        print('%s. output_grad_example'%self.optimizer.output_grad_example)
                    if hasattr(self.optimizer, 'output_example'):
                        print('%s. output_example'%self.optimizer.output_example)
                    if hasattr(self.optimizer, 'output_target_example'):
                        print('%s. output_target_example'%self.optimizer.output_target_example)
                    if hasattr(self.optimizer, 'output_truth_example'):
                        print('%s. output_truth_example'%self.optimizer.output_truth_example)
                    '''
                    if hasattr(self.optimizer, 'pred_error_rates'):
                        print('pred_error_rates: %s'%self.optimizer.pred_error_rates)
                    pass
            train_perform = self.model.get_perform(prefix='train: ', verbose=True)
            self.train_performs[self.epoch_now] = train_perform
            
            # evaluate model
            test_perform = self.optimizer.evaluate(test_data)
            '''
            self.model.reset_perform()
            for data in list(test_data):
                inputs, labels = data
                self.optimizer.evaluate({
                    'input': inputs.to(self.model.device), 
                    'output': labels.to(self.model.device)
                })
            '''
            self.test_performs[self.epoch_now] = test_perform
            if self.save_model and self.epoch_now%self.save_interval==1:
                self.model.save(save_path=self.save_model_path, save_name=self.model.dict['name'] + '_epoch=%d'%self.epoch_now)
            

            self.optimizer.update_epoch()
            self.epoch_now += 1
        
        if self.save_after_train:
            self.model.save(save_path=self.save_model_path, save_name=self.model.dict['name'] + '_epoch=afterTrain')
        
        self.plot_perform(save_path=self.anal_path)
    def plot_perform(self, save_path='./', save_name='perform.png', col_num=3):
        # plot test_performs
        epochs = self.test_performs.keys()
        epochs = np.array(epochs, )
        epochs = np.sort(epochs)
        items = self.test_performs[epochs[0]].keys()
        item_num = len(items)

        row_num = item_num // col_num
        if item_num % col_num > 0:
            col_num += 1
        
        fig, axes = plt.subplots(nrow=row_num, ncol=col_num)

        for item, index in enumerate(items):
            row_index = index // row_num
            col_index = index % row_num
            ax = axes[row_index, col_index]
            ax.set_title
        plt.suptitle('%s Test Performance'%self.model.dict['name'])
        ensure_path(save_path)
        plt.savefig(save_path + save_name)
        # plot train_performs
    
    def save(self, save_path, save_name=None):
        ensure_path(save_path)
        save_name = 'trainer' if save_name is None else save_name
        with open(save_path + save_name, 'wb') as f:
            torch.save(self.dict, f)
    def receive_options(self, options):
        self.options = options
        self.optimizer = self.options.optimizer
        self.model = self.options.model
        self.device = self.options.device
    def set_options(self):
        self.device = self.options.device
    def bind_model(self, model):
        self.model = model
    def detach_model(self):
        self.model = None
    def bind_optimizer(self, optimizer):
        self.optimizer = optimizer
    def detach_optimizer(self):
        self.optimizer = None
    def bind_data_loader(self, data_loader):
        self.data_loader = data_loader
    def detach_data_lader(self):
        self.data_loader = None

class Evaluator():
    def __init__(self, dict_={}, options=None):
        if options is not None:
            self.receive_options(options)
        self.dict = dict_
    def bind_data_loader(self, data_loader):
        self.data_loader = data_loader
    def bind_model(self, model):
        self.model = model
    '''
    def receive_options(self, options):
        self.options = options
        self.device = self.options.device
    '''
    def evaluate(self, model=None):
        model = self.model if model is None else model
        # evaluate model
        train_data, test_data = self.data_loader.get_data()
        model.reset_perform()
        for data in list(train_data):
            inputs, labels = data
            model.cal_perform(inputs.to(model.device), labels.to(model.device))
        test_perform = model.get_perform(prefix='test: ', verbose=True)