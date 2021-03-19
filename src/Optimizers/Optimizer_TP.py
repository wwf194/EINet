import torch

from utils import search_dict
from utils_model import build_optimizer, build_mlp

from Optimizers.Optimizer import *
from Optimizers.Optimizer import Optimizer

class Optimizer_TP(Optimizer):
    def __init__(self, dict_=None, load=False, options=None):
        super().__init__(dict_, load, options)
        self.dict = dict_
        self.mode = self.dict.setdefault('mode', 'train_on_r')
        if self.mode in ['train_on_r']:
            self.train = self.train_on_r
        elif self.mode in ['train_on_u']:
            self.train = self.train_on_u
        else:
            raise Exception('Optimizer_TP: Invalid mode: %s'%self.mode)

        if self.dict['loss_main'] in ['MSE', 'mse']:
            self.loss_func = torch.nn.MSELoss()
        elif self.dict['loss_main'] in ['CEL', 'cel']:
            self.loss_func = torch.nn.CrossEntropyLoss()
        else:
            raise Exception('Invalid main loss: %s'%self.loss_func)
    '''
    def receive_options(self, options):
        self.options = options
        self.device = options.device
        #self.model = options.model
        self.trainer = options.trainer
        #print('options.model:'+str(options.model))
        #self.build_optimizer()
    '''
    def bind_trainer(self, trainer):
        self.trainer = trainer
        self.build_optimizer()
        self.update_epoch_init()
        if self.scheduler is not None:
            self.dict['scheduler_dict'] = None
        else:
            self.dict['scheduler_dict'] = self.scheduler.state_dict()
        self.get_lr = self.get_current_lr
    def bind_model(self, model):
        self.model = model
    def update_before_train(self):
        #print(self.dict['update_before_train'])
        self.update_before_train_items = search_dict(self.dict, ['update_before_train'], default=[], write_default=True)
        
        for item in self.update_before_train_items:
            if item in ['alt_pc_act_strength', 'alt_pc_strength']:
                path = self.trainer.agent.walk_random(num=self.trainer.batch_size)
                self.model.alt_pc_act_strength(path)
            else:
                raise Exception('Invalid update_before_train item: %s'%str(item))
    def build_optimizer(self, load=False):
        self.optimizer = build_optimizer(self.dict['optimizer_forward'], model=self.model, load=load)
        self.optimizer_rec = build_optimizer(self.dict['optimizer_rec'], model=self.decoder_rec, load=load)
        self.optimizer_out = build_optimizer(self.dict['optimizer_out'], model=self.model_out, load=load)
    def build_decoder(self, load=False):

        self.lr_loss_func = self.build_decoder_
        self.decoder_out = self.build_decoder_single(self.dict['decoder_rec'], load=load)
        self.decoder_rec = self.build_decoder_single(self.dict['decoder_out'], load=load)
    def build_decoder_single(self, dict_, load=False):
        type_ = dict_['type']
        if type_ in ['mlp']:
            return build_mlp(dict_, load=load)
        else:
            raise Exception('Invalid decoder type: %s'%type_)
    def build_decoder_mlp(self):
        return
    def train_on_h(self, data, step_num=None, model=None):
        x, y = data['input'], ['output']
        if step_num is None:
            step_num = self.model.step_num
        if model is None:
            model = self.model
        
        self.optimizer.zero_grad()
        self.decoder_out.zero_grad()
        self.decoder_rec.zero_grad()

        # forward
        h_in_detach = torch.zeros([x.size(0), x.size(1)])
        
        h_in_detach_s = []
        h_out_s = []
        i_s = []
        for time in range(step_num):
            h_in_detach_s.append(h_in_detach)
            state = self.model.forward_once(x=x, h_in_detach=h_in_detach, detach_i=True, detach_u=False)
            i_s.append(state['i'])
            h_out_s.append(state['h_out'])
            h_in_detach = state['h_out'].detach() # for next iteration
            
        output = state['o']
        output_detach = state['o'].detach

        # train decoder_rec
        for time in range(step_num - 1):
            h_in_pred = self.decoder_rec(h_in_detach_s[time+1])
            loss_pred = self.lr_loss_func(h_in_pred, h_in_detach_s[time])
            loss_pred.backward(retain_graph=True)
        # train decoder_out
        h_in_pred = self.decoder_out(output_detach)
        loss_pred = self.lr_loss_func(h_in_pred, h_in_detach_s[step_num])
        loss_pred.backward(retain_graph=True)
        
        self.decoder_out.step()
        self.decoder_rec.step()

        # train model
        N_num = model.N_num
        output_truth = data['output']
        output_target = self.get_target(output_detach, output_truth).detach()
        loss = self.loss_func(output, output_target)
        loss.backward(retain_graph=True)
        # calculate target
        h_i_target = self.decoder_out(output_detach).detach()
        time = step_num - 2
        while(time>=0):
            loss_h = self.loss_func(h_i_target[:, 0:N_num], h_out_s[time])
            loss_h.backward(retain_graph=True)
            loss_i = self.loss_func(h_i_target[:, N_num:], i_s[time])
            loss_i.backward(retain_graph=True)
            h_i_target = self.decoder_rec(h_i_target).detach()
            time -= 1

        self.optimizer.zero_grad()
        loss = self.model.cal_perform(data)
        self.optimizer.step()

        model.cal_perform_from_output(output_detach, output_truth)
    def train_on_u(self, data):

        # forward

        # train decoder

        # train model
        return
    def evaluate(self, data):
        self.optimizer.zero_grad()
        self.model.reset_perform()
        loss = self.model.cal_perform(data)
        #self.model.get_perform(prefix='test', verbose=True)
        self.optimizer.zero_grad()
    def update_epoch_init(self): # decide what need to be done after every epoch 
        self.update_func_list = []
        self.update_func_list.append(self.update_lr_init())
    def update_epoch(self, **kw): # things to do after every epoch
        # in **kw: epoch_ratio: epoch_current / epoch_num_in_total
        for func in self.update_func_list:
            func(**kw)
    def update_lr_init(self): # define self.scheduler and return an update_lr method according to settings in self.dict_.
        #self.lr_decay = self.dict['lr_decay']
        lr_decay = self.lr_decay = self.dict['lr_decay']
        lr_decay_method = lr_decay.get('method')
        print(lr_decay_method)
        if lr_decay_method in ['None', 'none'] or lr_decay_method is None:
            return self.update_lr_none
        elif lr_decay_method in ['exp']:
            decay = search_dict(lr_decay, ['decay', 'coeff'], default=0.98, write_default=True, write_default_dict='decay')
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=decay)
            return self.update_lr
        elif lr_decay_method in ['stepLR', 'exp_interval']:
            decay = search_dict(lr_decay, ['decay', 'coeff'], default=0.98, write_default=True, write_default_key='decay')
            step_size = search_dict(lr_decay, ['interval', 'step_size'], default=0.98, write_default=True, write_default_key='decay')
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, step_size=step_size, gamma=decay)
            return self.update_lr
        elif lr_decay_method in ['Linear', 'linear']:
            milestones = search_dict(lr_decay, ['milestones'], throw_none_error=True)
            self.scheduler = LinearLR(self.optimizer, milestones=milestones, epoch_num=self.trainer.epoch_num)
            return self.update_lr
        else:
            raise Exception('Invalid lr decay method: '+str(lr_decay_method))
    def get_current_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']
    def update_lr(self, **kw):
        self.scheduler.step()
    def update_lr_none(self, **kw):
        return
    def detach_model(self):
        self.model = None