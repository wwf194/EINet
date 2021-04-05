import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import set_instance_attr, get_name, ensure_path, get_from_dict, search_dict
from utils_model import init_weight, cal_acc_from_label, get_loss_func, get_act_func, get_mask, get_ei_mask, get_cons_func

class RSLP(nn.Module):
    def __init__(self, dict_=None, load=False, f=None):
        super(RSLP, self).__init__()
        self.dict = dict_
        self.device = self.dict['device']
        
        #set_instance_attr(self, self.dict, exception=['N'])
        #self.N = Neurons_LIF(dict_ = self.dict['N'], load=load)
        #self.output_num = self.N.output_num
        
        # weight settings
        if load:
            #self.dict = torch.load(f, map_location=self.device) 
            self.i = self.dict['i'] # input weight
            self.register_parameter('i', self.i)
            self.i_b = self.dict['i_b'] # input bias
            if isinstance(self.i_b, torch.Tensor):
                self.register_parameter('i_b', self.i_b)
            self.o = self.dict['o'] # output weight
            self.register_parameter('o', self.o)
            self.r = self.dict['r'] # recurrent weight
            self.register_parameter('r', self.r)
            self.b = self.dict['r_b'] # recurrent bias
            if isinstance(self.i_b, torch.Tensor):
                self.register_parameter('i_b', self.i_b)            
            if self.dict['init_weight'] in ['nonzero']:
                self.h_init = self.dict['h_init'] # init hidden state
                self.register_parameter('h_init', self.h_init)
        else:
            self.i = torch.nn.Parameter(torch.zeros((self.dict['input_num'], self.dict['N_num']), device=self.device))
            init_weight(self.i, self.dict['init_weight']['i'])
            self.dict['i'] = self.i
            
            if self.dict['bias']:
                self.i_b = torch.nn.Parameter(torch.zeros((self.dict['N_num']), device=self.device))            
            else:
                self.i_b = 0.0
            self.dict['b_0'] = self.i_b

            self.dict['r_b'] = search_dict(self.dict, ['r_b', 'bias'], default=True, write_default=False)
            if self.dict['r_b']:
                self.r_b = torch.nn.Parameter(torch.zeros((self.dict['input_num']), device=self.device))
            else:
                self.b = 0.0
            self.dict['b'] = self.b
            self.o = self.dict['o'] = nn.Parameter(torch.zeros((self.dict['N_num'], self.dict['output_num']), device=self.device, requires_grad=True))
            self.r = self.dict['r'] = nn.Parameter(torch.zeros((self.dict['N_num'], self.dict['N_num']), device=self.device, requires_grad=True))
            
            if self.dict.get('init_weight') is None:
                self.dict['init_weight'] = {
                    'r': ['input', 1.0],
                    'o': ['input', 1.0]
                }

            init_weight(self.r, self.dict['init_weight']['r'])
            init_weight(self.o, self.dict['init_weight']['o'])    

        # set up input weight
        self.get_i = lambda :self.i

        # set up recurrent weight
        if self.dict['noself']:
            self.r_self_mask = torch.ones((self.dict['N_num'], self.dict['N_num']), device=self.device, requires_grad=False)
            for i in range(self.dict['N_num']):
                self.r_self_mask[i][i] = 0.0
            self.get_r_noself = lambda :self.r * self.r_self_mask
        else:
            self.get_r_noself = lambda :self.r
        self.ei_mask = None
        
        self.cons_func = get_cons_func(self.dict['cons_method'])
        if 'r' in self.dict['Dale']:
            self.ei_mask = get_ei_mask(E_num=self.dict['E_num'], N_num=self.dict['N_num']).to(self.device)
            self.get_r_ei = lambda :torch.mm(self.ei_mask, self.cons_func(self.get_r_noself()))
        else:
            self.get_r_ei = self.get_r_noself
        if 'r' in self.dict['mask']:
            self.r_mask = get_mask(N_num=self.dict['N_num'], output_num=self.dict['N_num']).to(self.device)
            self.get_r_mask = lambda :self.r_mask * self.get_r_ei()
        else:
            self.get_r_mask = self.get_r_ei
            
        self.get_r = self.get_r_mask

        # set up forward weight
        if 'o' in self.dict['Dale']: #set mask for EI separation
            if(self.ei_mask is None):
                self.ei_mask = get_ei_mask(E_num=self.dict['E_num'], N_num=self.dict['N_num'])
            self.get_o_ei = lambda :torch.mm(self.ei_mask, self.cons_func(self.o))
        else:
            self.get_o_ei = lambda :self.o
        if 'o' in self.dict['mask']: #set mask for connection pruning
            self.o_mask = get_mask(N_num=self.dict['N_num'], output_num=self.dict['output_num'])
            self.get_o_mask = lambda :self.o_mask * self.get_o_ei()
        else:
            self.get_o_mask = self.get_o_ei            
        self.get_o = self.get_o_mask

        self.N_num = self.dict['N_num']
        if self.dict['separate_ei']:
            self.time_const_e = self.dict['time_const_e']
            self.time_const_i = self.dict['time_const_i']
            self.act_func = self.act_func_ei
            self.act_func_e = get_act_func(self.dict['act_func_e'])
            self.act_func_i = get_act_func(self.dict['act_func_i'])
            self.E_num = self.dict['E_num']
            self.I_num = self.dict['I_num']
            self.cal_x = self.cal_x_ei
            self.get_weight = self.get_weight_ei
            self.response = self.response_ei
            self.update_weight_cache = self.update_weight_cache_ei
            self.weight_names = ['E->E','E->I','I->E','I->I','E->Y','I->Y','N->N','E.r','E.l','I.r','I.l','E.b','I.b','r','E.f','I.f','b']
        else:
            self.time_const = self.dict['time_const']
            self.act_func = get_act_func(self.dict['act_func'])
            self.cal_x = self.cal_x_uni
            self.get_weight = self.get_weight_uni
            self.response = self.response_uni
            self.weight_names = ['N->Y','N->N','N.f','o','b','r']

        # loss settings
        self.loss_dict = self.dict['loss']

        self.main_loss_func = get_loss_func(self.loss_dict['main_loss'], truth_is_label=True, num_class=self.loss_dict['num_class'])
        '''
        if self.loss_dict['main_loss'] in ['CEL', 'cel']:
            self.main_loss_func = torch.nn.CrossEntropyLoss()
        elif self.loss_dict['main_loss'] in ['MSE', 'mse']:
            self.main_loss_func = torch.nn.MSELoss()
        '''

        input_mode = get_name(self.dict['input_mode'])
        if input_mode in ['endure'] or input_mode is None: #default
            self.prep_input = self.prep_input_endure
            self.get_input = self.get_input_endure

        self.main_loss_coeff = self.loss_dict['main_loss_coeff']

        self.perform_list = {'class':0.0, 'act':0.0, 'weight':0.0, 'acc':0.0}

        self.hebb_coeff = self.loss_dict.setdefault('hebb_coeff', 0.0)
        self.act_coeff = self.loss_dict.setdefault('act_coeff', 0.0)
        self.weight_coeff = self.loss_dict.setdefault('weight_coeff', 0.0)
        if self.hebb_coeff != 0.0:
            self.perform_list['hebb'] = 0.0
        self.batch_count = 0
        self.sample_count = 0
        self.step_num = self.dict['step_num']

        if self.dict['separate_ei']:
            self.get_weight = self.get_weight_ei
            self.update_weight_cache = self.update_weight_cache_ei
            self.response_keys = ['E.u','E.x','I.u','I.x','E->E','E->I','I->E','I->I','E->Y','I->Y', 'X->E', 'X->I', 'N->Y', 'N->N', 'u']
            self.weight_names = ['X->E', 'X->I', 'i']
            self.E_num = self.dict['E_num']
            self.I_num = self.dict['I_num']
            self.N_num = self.dict['N_num']
        else:
            self.response_keys = ['o','r','u']
            self.get_weight = self.get_weight_uni
            self.N_num = self.dict['N_num']
        self.cache = {}

    def prep_input_once(self, i):
        #i = i.to(self.device)
        return torch.mm(i, self.get_i()) + self.i_b
    def prep_input_once_no_grad(self, i_):
        return torch.mm(i_, self.get_i().detach()) + self.i_b

    def prep_input_full(self, i_):
        i_ = torch.mm(i_, self.get_i()) + self.i_b # [batch_size, N_num]
        i_unsqueezed = torch.unsqueeze(i_, 1)
        return torch.cat([i_unsqueezed for _ in range(self.step_num)], dim=1) # [batch_size, step_num, N_num]
    def prep_input_endure(self, x): # x: [batch_size, input_num]
        i = torch.mm(x, self.get_i()) + self.i_b # [batch_size, N_num]
        self.cache['input'] = i
    def get_input_endure(self, time=None):
        return self.cache['input']
    def forward(self, x, step_num=None): # [batch_size, C x H x W]
        if step_num is None:
            step_num = self.step_num
        act_list = []
        output_list = []

        x = x.view(x.size(0), -1)
        
        self.prep_input(x)
        #self.N.reset_x(batch_size=x.size(0))
        #i_ = self.prep_input(x) # [step_num, batch_size, N_num]

        x = None
        h = None
        for time in range(step_num):
            state = self.forward_once(x=x, i=self.get_input(time), h=h)
            x, u, h, o = get_from_dict(state, ['x', 'u', 'h', 'o'])

            act_list.append(u) # [batch_size, N_num]
            output_list.append(o) # [batch_size, output_num]
        
        output_list = list(map(lambda x:torch.unsqueeze(x, 1), output_list))
        act_list = list(map(lambda x:torch.unsqueeze(x, 1), act_list))
        output = torch.cat(output_list, dim=1) # [batch_size, step_num, output_num]
        act = torch.cat(act_list, dim=1) # [batch_size, step_num, N_num]
        return {
            'output': output, 
            'act': act
        }
    def reset(self, **kw):
        self.N.reset_x(**kw)
    '''
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
    '''
    def forward_once_no_grad(self, x=None, i=None, h_in_detach, detach_i=False, detach_u=False, reset=False): # [batch_size, input_num]
        if reset:
            self.N.reset()
        
        x = x.view(x.size(0), -1)
        i = self.prep_input_once_no_grad(x)
        if detach_i:
            i_ = i.detach()        
        else:
            i_ = i
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
    
    def forward(self, i):
        dx = i + self.b
        self.x = self.cal_x(dx) #x: [batch_size, N_num]
        u = self.act_func(self.x)
        o = u.mm(self.get_o())
        h = u.mm(self.get_r())
        return o, h, u
    def forward_layer_once(self, x=None, h=None, i=None, detach_u=False):
        if x is None: # first iteration
            x = self.get_init_state(batch_size=i.size(0))
        dx = i + self.b
        x = self.cal_x(dx, x=x) #x: [batch_size, N_num]
        u = self.act_func(x)
        if detach_u:
            u_detach = u.detach()
            o = u_detach.mm(self.get_o())
            h = u_detach.mm(self.get_r())
            return {
                'u_detach': u_detach,
                'x': x,
                'h': h,
                'o': o,
            }
        else:
            o = u.mm(self.get_o())
            h = u.mm(self.get_r())
            return {
                'u': u,
                'x': x,
                'h': h,
                'o': o,
            }
    def forward_once_no_grad(self, x, h, i, detach_u=False):
        dx = i + self.b
        x = self.cal_x(dx, x=x) #x: [batch_size, N_num]
        u = self.act_func(x)
        if detach_u:
            u_detach = u.detach()
            o = u_detach.mm(self.get_o().detach())
            h = u_detach.mm(self.get_r().detach())
            return {
                'u_detach': u_detach,
                'x': x,
                'h': h,
                'o': o,
            }
        else:
            o = u.mm(self.get_o().detach())
            h = u.mm(self.get_r().detach())
            return {
                'u': u,
                'x': x,
                'h': h,
                'o': o,
            }
    def get_init_state_zero(self, **kw):
            batch_size = kw['batch_size']
            return torch.zeros(, device=self.device)
    def reset_x_zero(self, **kw):
        #print(batch_size)
        self.x = torch.zeros((kw['batch_size'], self.dict['N_num']), device=self.device) #(batch_size, input_num)
    def get_noise_gaussian(self, batch_size, N_num):
        noise = torch.zeros((batch_size, N_num), device=self.device)
        torch.nn.init.normal_(noise, 0.0, self.noise_coeff)
        return noise
    def act_func_ei(self, x):
        return torch.cat( [self.act_func_e(x[:, 0:self.E_num]), self.act_func_i(x[:, self.E_num:self.N_num])], dim=1)
    def cal_x_uni(self, dx, x=None):
        if x is None:
            x = self.x
        return (1.0 - self.time_const) * (x + self.get_noise(dx.size(0), self.N_num)) + self.time_const * dx #x: [batch_size, N_num]
    def cal_x_ei(self, dx, x=None):
        if x is None:
            x = self.x
        x_e = (1.0 - self.time_const_e) * (x[:, 0:self.E_num] + self.get_noise(dx.size(0), self.E_num)) + self.time_const_e * dx[:, 0:self.E_num] # x: [batch_size, E_num]
        x_i = (1.0 - self.time_const_i) * (x[:, self.E_num:self.N_num] + self.get_noise(dx.size(0), self.I_num)) + self.time_const_i * dx[:, self.E_num:self.N_num] # x: [batch_size, I_num]        
        return torch.cat([x_e, x_i], dim=1)
    def response(self, x, step_num=None):
        if step_num is None:
            step_num = self.step_num
        x = x.view(x.size(0), -1)
        self.N.reset_x(batch_size=x.size(0))
        i_ = self.prep_input_full(x) #(batch_size, step_num, N_num)
        r = 0.0
        for time in range(step_num):
            i_tot = torch.squeeze(i_[:, time, :]) + r
            f, r, u, res = self.N.response(i_tot)
        if self.dict['separate_ei']:
            res_X = torch.squeeze(i_[:, -1, :])
            res['X->E'] = res_X[:, 0:self.E_num]
            res['X->I'] = res_X[:, self.E_num:self.N_num]
        else:
            res['X->N'] = torch.squeeze(i_[:, -1, :])#(batch_size, N_num)
        return res
    def iter(self, x, step_num=None, to_cpu_interval=10):
        if step_num is None:
            step_num = self.step_num
        x = x.view(x.size(0), -1)
        self.N.reset_x(batch_size=x.size(0))
        i_ = self.prep_input_full(x) #(batch_size, step_num, N_num)
        ress = {} #responses
        ress_cat = {}
        keys = self.response_keys
        for key in keys:
            ress[key] = []
            ress_cat[key] = None
        r = 0.0
        for time in range(step_num):
            i_tot = torch.squeeze(i_[:, time, :]) + r
            f, r, u, res = self.N.response(i_tot)
            for key in res.keys():
                ress[key].append(res[key]) #[key](batch_size, unit_num)
            if((time+1)%to_cpu_interval == 0): #avoid GPU OOM.
                cat_dict(ress_cat, ress, dim_unsqueeze=1, dim=1)
        cat_dict(ress_cat, ress, dim_unsqueeze=1, dim=1) #cat along step_num dim.
        if self.dict['separate_ei']:
            ress_cat['X->E'] = i_[:, :, 0:self.E_num]
            ress_cat['X->I'] = i_[:, :, self.E_num:self.N_num]
        else:
            ress_cat['X->N'] = i_
        return ress_cat #(batch_size, step_num, N_num)
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
        if self.cache.get('o') is not None:
            f_0 = self.cache['o']
            f_change_rate = np.sum(abs(f_1 - f_0)) / np.sum(np.abs(f_0))
            result += 'f_change_rate: %.3f '%f_change_rate
        self.cache['o'] = f_1

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
                w = self.i_b
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
    def get_iter_data(self, data_loader, step_num=None, batch_num=None):
        print('calculating iter_data. batch_num=%d'%(len(data_loader)))
        if(step_num is None):
            step_num = self.step_num
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

        iter_data['acc'] = [0.0 for _ in range(step_num)]
        iter_data['loss'] = [0.0 for _ in range(step_num)]
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
            res=self.iter(inputs) #[key](batch_size, step_num, unit_num)
            for key in ress.keys():
                ress[key].append(res[key])
            
            for time in range(step_num):
                iter_data['loss'][time] += self.main_loss_func( torch.squeeze(res['N->Y'][:,time,:]), labels)
                iter_data['acc'][time] += ( torch.max( torch.squeeze(res['N->Y'][:,time,:] ), 1)[1]==labels).sum().item()

        for time in range(step_num):
            iter_data['loss'][time] = iter_data['loss'][time] / count
            iter_data['acc'][time] = iter_data['acc'][time] / label_count
        
        cat_dict(iter_data, ress, dim=0) #cat along batch_size dim.         
        self.train()

        #print(len(iter_data['loss']))
        #input()

        return iter_data
    def get_res_data(self, data_loader, step_num=None, batch_num=None):
        print('calculating res_data. batch_num=%d'%(len(data_loader)))
        if(step_num is None):
            step_num = self.step_num
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
        self.cache['o'] = self.get_f().detach().cpu().numpy()
    def report_weight_update(self):
        i = self.get_i().detach().cpu().numpy()
        r = self.get_r().detach().cpu().numpy()
        f = self.get_f().detach().cpu().numpy()

        i_delta = np.sum(np.abs(i - self.cache['i'])) / np.sum(np.abs(self.cache['i']))
        r_delta = np.sum(np.abs(r - self.cache['r'])) / np.sum(np.abs(self.cache['r']))
        f_delta = np.sum(np.abs(f - self.cache['o'])) / np.sum(np.abs(self.cache['o']))
        print('weight update rate: i:%.4e r:%.4e f:%.4e'%(i_delta, r_delta, f_delta))


    def response_uni(self, i_):
        res = {}
        dx = i_ + self.b
        self.x = self.cal_x(dx) #x: [batch_size, N_num]
        res['x'] = self.x
        u = self.act_func(self.x)
        res['u'] = u
        res['o'] = u.mm(self.get_o())
        res['r'] = u.mm(self.get_r())
        return res
    def response_ei(self, i_):
        res = {}
        dx = i_ + self.b
        self.x = self.cal_x(dx) #x: [batch_size, N_num]
        u = self.act_func(self.x)
        res['u'] = u

        res['E.x'] = self.x[:, 0:self.E_num]
        res['I.x'] = self.x[:, self.E_num:self.N_num]
        res['E.u'] = u[:, 0:self.E_num]
        res['I.u'] = u[:, self.E_num:self.N_num]        

        res['E->E'] = torch.mm(res['E.u'], self.cache['weight_cache']['E->E'])
        res['E->I'] = torch.mm(res['E.u'], self.cache['weight_cache']['E->I'])
        res['I->E'] = torch.mm(res['I.u'], self.cache['weight_cache']['I->E'])
        res['I->I'] = torch.mm(res['I.u'], self.cache['weight_cache']['I->I'])
        
        f = u.mm(self.get_o())
        r = u.mm(self.get_r())
        res['N->Y'] = f
        res['N->N'] = r

        res['E->Y'] = torch.mm(res['E.u'], self.cache['weight_cache']['E->Y'])
        res['I->Y'] = torch.mm(res['I.u'], self.cache['weight_cache']['I->Y'])

        return f, r, u, res
    def get_weight_ei(self, name, positive=True, detach=False):
        sig_r = False
        sig_f = False
        if name in ['E.r', 'E->E']:
            w = self.get_r()[0:self.E_num, 0:self.E_num]
        elif name in ['I.r', 'I->I']:
            w = self.get_r()[self.E_num:self.N_num, self.E_num:self.N_num]
            sig_r = True
        elif name in ['E.l', 'E->I']:
            w = self.get_r()[0:self.E_num, self.E_num:self.N_num]
        elif name in ['I.l', 'I->E']:
            w = self.get_r()[self.E_num:self.N_num, 0:self.E_num]
            sig_r = True
        elif name in ['E.f', 'E->Y']:
            w = self.get_o()[0:self.E_num, :]
        elif name in ['I.f', 'I->Y']:
            w = self.get_o()[self.E_num:self.N_num, :]
            sig_f = True
        elif name in ['b']:
            w = self.b
        elif name in ['E.b']:
            if(isinstance(self.b, float)):
                return self.b
            else:
                w = self.b[0:self.E_num]
        elif name in 'I.b':
            if(isinstance(self.b, float)):
                return self.b
            else:
                w = self.b[self.E_num:self.N_num]
        elif name in ['r', 'N->N']:
            #print('getting weight r')
            w = self.get_r()
            #print(w)
            if positive:
                w = torch.abs(w)
        elif name in ['o']:
            w = self.get_o()
        else:
            return 'invalid weight name:%s'%(name)
        if detach:
            w = w.detach()
        if(positive and sig_r and ('r' in self.dict['Dale'])):
            w = - w
        elif(positive and sig_f and ('o' in self.dict['Dale'])):
            w = - w
    
        return w
    def get_weight_uni(self, name, positive=None):
        if name in ['r', 'N->N']:
            return self.get_r()
        elif name in ['b']:
            return self.get_b()
        else:
            raise Exception('Invalid weight name: %s'%name)
    def update_weight_cache_ei(self):
        self.cache['weight_cache'] = {}
        weight_cache = self.cache['weight_cache']
        N_r = self.get_weight('r', positive=True)
        weight_cache['E->E'] = N_r[0:self.E_num, 0:self.E_num]
        weight_cache['I->I'] = N_r[self.E_num:self.N_num, self.E_num:self.N_num]
        weight_cache['E->I'] = N_r[0:self.E_num, self.E_num:self.N_num]
        weight_cache['I->E'] = N_r[self.E_num:self.N_num, 0:self.E_num]
        N_f = self.get_weight('o')
        weight_cache['E->Y'] = N_f[0:self.E_num, :]
        weight_cache['I->Y'] = N_f[self.E_num:self.N_num, :]