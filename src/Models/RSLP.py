import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import set_instance_attr, get_name, ensure_path
from utils_model import init_weight, cal_acc_from_label, get_loss_func

from Models.Neurons_LIF import Neurons_LIF

class RSLP(nn.Module):
    def __init__(self, dict_=None, load=False, f=None):
        super(RSLP, self).__init__()
        self.dict = dict_
        #self.device = self.dict['device']
        set_instance_attr(self, self.dict, exception=['N'])
        self.N = Neurons_LIF(dict_ = self.dict['N'], load=load)
        self.output_num = self.N.output_num
        # set up weights
        if load:
            self.dict=torch.load(f, map_location=self.device) 
            self.i = self.dict['i'] #input weight
            self.b_0 = self.dict['b_0']
        else:
            self.i = torch.nn.Parameter(torch.zeros((self.dict['input_num'], self.dict['N_num']), device=self.device))
            init_weight(self.i, self.dict['init_weight']['i'])
            self.dict['i'] = self.i
            
            if self.dict['bias']:
                self.b_0 = torch.nn.Parameter(torch.zeros((self.dict['N_num']), device=self.device))            
            else:
                self.b_0 = 0.0
            self.dict['b_0'] = self.b_0

        self.get_i = lambda :self.i
        self.get_f = self.N.get_f
        self.get_r = self.N.get_r

        input_mode = get_name(self.dict['input_mode'])
        if input_mode in ['endure'] or input_mode is None: #default
            self.prep_input = self.prep_input_endure
            self.get_input = self.get_input_endure

        self.loss_dict = self.dict['loss']

        self.main_loss_func = get_loss_func(self.loss_dict['main_loss'], truth_is_label=True, num_class=self.loss_dict['num_class'])
        '''
        if self.loss_dict['main_loss'] in ['CEL', 'cel']:
            self.main_loss_func = torch.nn.CrossEntropyLoss()
        elif self.loss_dict['main_loss'] in ['MSE', 'mse']:
            self.main_loss_func = torch.nn.MSELoss()
        '''
        self.main_loss_coeff = self.loss_dict['main_loss_coeff']

        self.perform_list = {'class':0.0, 'act':0.0, 'weight':0.0, 'acc':0.0}

        self.hebb_coeff = self.loss_dict.setdefault('hebb_coeff', 0.0)
        self.act_coeff = self.loss_dict.setdefault('act_coeff', 0.0)
        self.weight_coeff = self.loss_dict.setdefault('weight_coeff', 0.0)
        if self.hebb_coeff != 0.0:
            self.perform_list['hebb'] = 0.0
        self.batch_count = 0
        self.sample_count = 0
        self.iter_time = self.dict['iter_time']

        if self.dict['separate_ei']:
            self.get_weight = self.get_weight_ei
            self.update_weight_cache = self.update_weight_cache_ei
            self.response_keys = ['E.u','E.x','I.u','I.x','E->E','E->I','I->E','I->I','E->Y','I->Y', 'X->E', 'X->I', 'N->Y', 'N->N', 'u']
            self.weight_names = ['X->E', 'X->I', 'i']
        else:
            self.response_keys = ['f','r','u']
            self.get_weight = self.get_weight_uni
        self.cache = {}
        self.E_num = self.N.dict['E_num']
        self.I_num = self.N.dict['I_num']
        self.N_num = self.N.dict['N_num']
        self.dict['noself'] = self.N.dict['noself']

        self.cache = {}
    def prep_input_once(self, i_):
        return torch.mm(i_, self.get_i()) + self.b_0
    def prep_input_endure(self, i_):
        i_ = torch.mm(i_, self.get_i()) + self.b_0 #(batch_size, N_num)
        self.cache['input'] = i_
    def prep_input_full(self, i_):
        i_ = torch.mm(i_, self.get_i()) + self.b_0 #(batch_size, N_num)
        i_unsqueezed = torch.unsqueeze(i_, 1)
        return torch.cat([i_unsqueezed for _ in range(self.iter_time)], dim=1) #(batch_size, iter_time, N_num)        
    def get_input_endure(self, time=None):
        return self.cache['input']
    def forward(self, x, iter_time=None): #(batch_size, pixel_num)
        if(iter_time is None):
            iter_time = self.iter_time
        x = x.view(x.size(0), -1)
        self.N.reset_x(batch_size=x.size(0))
        i_ = self.prep_input(x) #(iter_time, batch_size, N_num)
        act_list = []
        output_list = []
        r = 0.0
        for time in range(iter_time):
            f, r, u = self.N.forward(torch.squeeze(self.get_input(time)) + r)
            act_list.append(u) #(batch_size, N_num)
            output_list.append(f) #(batch_size, output_num)
        output_list = list(map(lambda x:torch.unsqueeze(x, 1), output_list))
        act_list = list(map(lambda x:torch.unsqueeze(x, 1), act_list))
        output = torch.cat(output_list, dim=1) #(batch_size, iter_time, output_num)
        act = torch.cat(act_list, dim=1) #(batch_size, iter_time, N_num)
        return output, act
    def reset(self, **kw):
        self.N.reset_x(**kw)
    def forward_once(self, x, h_in_detach, detach_i=True, detach_u=False, reset=False): # [batch_size, input_num]
        if reset:
            self.N.reset()
        #print(x.device) 
        x = x.view(x.size(0), -1)
        i = self.prep_input_once(x)
        if detach_i:
            i_ = i.detach()        
        else:
            i_ = i
        #print(i_.device)
        #print(h_in_detach.device)
        o, h_out, u = self.N.forward_once(i_ + h_in_detach, detach_u=detach_u)
        state = {
            'o': o,
            'h_out': h_out,
            'i': i,
        }
        if detach_i:
            state['i_detach'] = i_
        if detach_u:
            state['u_detach'] = u
        else:
            state['u'] = u
        return state
    def response(self, x, iter_time=None):
        if(iter_time is None):
            iter_time = self.iter_time
        x = x.view(x.size(0), -1)
        self.N.reset_x(batch_size=x.size(0))
        i_ = self.prep_input_full(x) #(batch_size, iter_time, N_num)
        r = 0.0
        for time in range(iter_time):
            i_tot = torch.squeeze(i_[:, time, :]) + r
            f, r, u, res = self.N.response(i_tot)
        if self.dict['separate_ei']:
            res_X = torch.squeeze(i_[:, -1, :])
            res['X->E'] = res_X[:, 0:self.E_num]
            res['X->I'] = res_X[:, self.E_num:self.N_num]
        else:
            res['X->N'] = torch.squeeze(i_[:, -1, :])#(batch_size, N_num)
        return res
    def iter(self, x, iter_time=None, to_cpu_interval=10):
        if(iter_time is None):
            iter_time = self.iter_time
        x = x.view(x.size(0), -1)
        self.N.reset_x(batch_size=x.size(0))
        i_ = self.prep_input_full(x) #(batch_size, iter_time, N_num)
        ress = {} #responses
        ress_cat = {}
        keys = self.response_keys
        for key in keys:
            ress[key] = []
            ress_cat[key] = None
        r = 0.0
        for time in range(iter_time):
            i_tot = torch.squeeze(i_[:, time, :]) + r
            f, r, u, res = self.N.response(i_tot)
            for key in res.keys():
                ress[key].append(res[key]) #[key](batch_size, unit_num)
            if((time+1)%to_cpu_interval == 0): #avoid GPU OOM.
                cat_dict(ress_cat, ress, dim_unsqueeze=1, dim=1)
        cat_dict(ress_cat, ress, dim_unsqueeze=1, dim=1) #cat along iter_time dim.
        if self.dict['separate_ei']:
            ress_cat['X->E'] = i_[:, :, 0:self.E_num]
            ress_cat['X->I'] = i_[:, :, self.E_num:self.N_num]
        else:
            ress_cat['X->N'] = i_
        return ress_cat #(batch_size, iter_time, N_num)
    def get_perform(self, prefix=None, verbose=True):
        perform_str = ''
        if prefix is not None:
            perform_str += prefix
        for key in self.perform_list.keys():
            if key in ['acc']:
                if self.sample_count > 0:
                    perform_str += '%s:%.4f '%(key, self.perform_list[key]/self.sample_count)
                else:
                    perform_str += '%s:no_log'%(key)
            else:
                if self.batch_count > 0:
                    #print('%s:%s'%(key, str(self.perform_list[key]/self.batch_count)), end=' ')
                    perform_str += '%s:%.4e '%(key, self.perform_list[key]/self.batch_count)
                else:
                    perform_str += '%s:no_log '%(key)
        if verbose:
            print(perform_str)
        return perform_str
    def reset_perform(self):
        self.batch_count = 0
        self.sample_count = 0
        #print(self.batch_count)
        for key in self.perform_list.keys():
            self.perform_list[key] = 0.0
    def cal_perform(self, data):
        x, y = data['input'].to(self.device), data['output'].to(self.device)
        #x: [batch_size, step_num, input_num]
        #y: [batch_size, step_num, output_num]
        output, act = self.forward(x)
        #self.dict['act_avg'] = torch.mean(torch.abs(act))
        #print(output.size())
        #input()
        return self.cal_perform_from_output(output, y, act)
    def cal_perform_from_output(self, output, output_truth, act=None):
        if len(list(output.size()))==3:
            output = output[:, -1, :]
        #print(output.size())
        #print(output_truth.size())
        loss_class = self.main_loss_coeff * self.main_loss_func(output, output_truth)
        if act is None:
            loss_act = torch.zeros([1], device=self.device)
        else:
            loss_act = self.act_coeff * torch.mean(act ** 2)

        loss_weight = self.weight_coeff * ( torch.mean(self.N.get_r() ** 2) )
        self.perform_list['weight'] = self.perform_list['weight'] + loss_weight.item()
        self.perform_list['act'] = self.perform_list['act'] + loss_act.item()
        self.perform_list['class'] = self.perform_list['class'] + loss_class.item()
        correct_num, sample_num = cal_acc_from_label(output, output_truth) 
        self.perform_list['acc'] += correct_num
        self.batch_count += 1
        self.sample_count += output.size(0)
        #print(self.batch_count)
        if self.hebb_coeff==0.0:
            loss_hebb = 0.0 
        else:
            loss_hebb = self.cal_perform_hebb(act)
        return loss_class + loss_act + loss_weight + loss_hebb
    def anal_weight_change(self, verbose=True):
        result = ''
        r_1 = self.get_r().detach().cpu().numpy()
        if self.cache.get('r') is not None:
            r_0 = self.cache['r']
            r_change_rate = np.sum(abs(r_1 - r_0)) / np.sum(np.abs(r_0))
            result += 'r_change_rate: %.3f '%r_change_rate
        self.cache['r'] = r_1

        f_1 = self.get_f().detach().cpu().numpy()
        if self.cache.get('f') is not None:
            f_0 = self.cache['f']
            f_change_rate = np.sum(abs(f_1 - f_0)) / np.sum(np.abs(f_0))
            result += 'f_change_rate: %.3f '%f_change_rate
        self.cache['f'] = f_1

        if hasattr(self, 'get_i'):
            i_1 = self.get_i().detach().cpu().numpy()
            if self.cache.get('i') is not None:
                i_0 = self.cache['i']
                i_change_rate = np.sum(abs(i_1 - i_0)) / np.sum(np.abs(i_0))
                result += 'i_change_rate: %.3f '%i_change_rate
            self.cache['i'] = i_1
        if verbose:
            print(result)
        return result
    def cal_perform_hebb(self, act):
        x = torch.squeeze(act[-1, :, :]) # [batch_size, N_num]
        batch_size=x.size(1)
        x = x.detach().cpu().numpy()
        x = torch.from_numpy(x).to(self.device)
        weight=self.N.get_r() # [N_num, N_num]
        act_var = torch.var(x, dim=0) # [N_num]
        act_var = act_var * (batch_size - 1)
        act_mean = torch.mean(x, dim=0).detach() # [N_num]
        #convert from tensor and numpy prevents including the process of computing var and mean into the computation graph.
        #act_std = torch.from_numpy(act_var).to(self.device) ** 0.5
        #act_mean = torch.from_numpy(act_mean).to(self.device)
        act_std = act_var ** 0.5
        std_divider = torch.mm(torch.unsqueeze(act_std, 1), torch.unsqueeze(act_std, 0)) # [N_num, N_num]
        
        x_normed = (x - act_mean) #broadcast
        act_dot = torch.mm(x_normed.t(), x_normed)
        try:
            act_corr = act_dot / std_divider
        except Exception:
            abnormal_coords = []
            for i in range(list(act_std.size())[0]):
                for j in range(list(act_std.size()[1])):
                    if(std_divider[i][j] == 0.0):
                        print('act_std[%d]][%d]==0.0'%(i, j))
                        std_divider[i][j] = 1.0
                        abnormal_coords.append([i,j])
            act_corr = act_dot / std_divider
            for coord in abnormal_coords:
                act_corr[coord[0]][coord[1]] = 0.0
        
        act_corr = act_corr.detach().cpu().numpy()
        act_corr = torch.from_numpy(act_corr).to(self.device)
        
        #print(weight.self.device)
        #print(act_corr.self.device)
        #if self.training:
            #print(act_corr)
            #print('act_corr')
            #input()
        self.dict['last_act_corr'] = act_corr.detach().cpu()
        return - self.hebb_coeff * torch.mean(torch.tanh(torch.abs(weight)) * act_corr)
    def save(self, save_path, save_name):   
        ensure_path(save_path)
        with open(save_path + save_name, 'wb') as f:
            net = self.to(torch.device('cpu'))
            torch.save(net.dict, f)
            net = self.to(self.device)
    def get_weight_ei(self, name, detach=False, positive=True):
        if name in self.N.weight_names:
            return self.N.get_weight(name=name, detach=detach, positive=positive)
        elif name in self.weight_names:
            if name in ['b_0']:
                w = self.b_0
            elif name in ['i']:
                w = self.get_i()
            elif name in ['X->E']:
                w = self.get_i()[:, 0:self.E_num]
            elif name in ['X->I']:
                w = self.get_i()[:, self.E_num:self.N_num]
        else:
            print('invalid weight name:%s'%(name))
        if detach:
            w = w.detach()
        return w
    def update_weight_cache_ei(self):
        self.N.update_weight_cache_ei()
        self.cache['weight_cache'] = self.N.cache['weight_cache']
        #print('E->E: ')
        #print(self.cache['weight_cache']['E->E'])
        #print(self.N.get_weight('E->E'))
        #input()
    def get_iter_data(self, data_loader, iter_time=None, batch_num=None):
        print('calculating iter_data. batch_num=%d'%(len(data_loader)))
        if(iter_time is None):
            iter_time = self.iter_time
        self.update_weight_cache()
        self.eval()
        count=0
        data_loader = list(data_loader)
        if(batch_num is None):
            batch_num = len(data_loader)
        else:
            data_loader = random.sample(data_loader, 50)
        ress = {}
        iter_data = {}
        keys = self.response_keys
        for key in keys:
            iter_data[key] = None
            ress[key] = []

        iter_data['acc'] = [0.0 for _ in range(iter_time)]
        iter_data['loss'] = [0.0 for _ in range(iter_time)]
        #print('aaa')
        #print(len(iter_data['acc']))
        #print(len(iter_data['loss'])) 
        label_count = 0
        for data in data_loader:
            inputs, labels = data
            inputs=inputs.to(self.device)
            #labels=labels.to(self.device)
            count += 1
            label_count += labels.size(0)
            res=self.iter(inputs) #[key](batch_size, iter_time, unit_num)
            for key in ress.keys():
                ress[key].append(res[key])
            
            for time in range(iter_time):
                iter_data['loss'][time] += self.main_loss_func( torch.squeeze(res['N->Y'][:,time,:]), labels)
                iter_data['acc'][time] += ( torch.max( torch.squeeze(res['N->Y'][:,time,:] ), 1)[1]==labels).sum().item()

        for time in range(iter_time):
            iter_data['loss'][time] = iter_data['loss'][time] / count
            iter_data['acc'][time] = iter_data['acc'][time] / label_count
        
        cat_dict(iter_data, ress, dim=0) #cat along batch_size dim.         
        self.train()

        #print(len(iter_data['loss']))
        #input()

        return iter_data
    def get_res_data(self, data_loader, iter_time=None, batch_num=None):
        print('calculating res_data. batch_num=%d'%(len(data_loader)))
        if(iter_time is None):
            iter_time = self.iter_time
        self.update_weight_cache()
        self.eval()
        count=0
        data_loader = list(data_loader)
        if(batch_num is None):
            batch_num = len(data_loader)
        else:
            data_loader = random.sample(data_loader, 50)
        ress = {}
        res_data = {}
        keys = self.response_keys
        for key in keys:
            res_data[key] = None
            ress[key] = []
        for data in data_loader:
            count=count+1
            inputs, labels = data
            inputs = inputs.to(self.device)
            #labels=labels.to(device)
            res=self.response(inputs) #[key](batch_size, unit_num)
            for key in res.keys():
                ress[key].append(res[key])
        cat_dict(res_data, ress, dim=0) #cat along batch_size dim.         
        self.train()
        return res_data
    def print_weight_info(self):
        ei = self.get_weight('E->I', positive=True)
        ie = self.get_weight('I->E', positive=True)
        er = self.get_weight('E->E', positive=True)
        ir = self.get_weight('I->I', positive=True)
        weights = [ei, ie, er, ir]
        for w in weights:
            print(w)
            print(torch.mean(w))
            print(torch.min(w))
            print(torch.max(w))
            print(list(w.size()))

    def cache_weight(self):
        self.cache['i'] = self.get_i().detach().cpu().numpy()
        self.cache['r'] = self.get_r().detach().cpu().numpy()
        self.cache['f'] = self.get_f().detach().cpu().numpy()
    def report_weight_update(self):
        i = self.get_i().detach().cpu().numpy()
        r = self.get_r().detach().cpu().numpy()
        f = self.get_f().detach().cpu().numpy()

        i_delta = np.sum(np.abs(i - self.cache['i'])) / np.sum(np.abs(self.cache['i']))
        r_delta = np.sum(np.abs(r - self.cache['r'])) / np.sum(np.abs(self.cache['r']))
        f_delta = np.sum(np.abs(f - self.cache['f'])) / np.sum(np.abs(self.cache['f']))
        print('weight update rate: i:%.4e r:%.4e f:%.4e'%(i_delta, r_delta, f_delta))