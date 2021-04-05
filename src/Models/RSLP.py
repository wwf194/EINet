import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import set_instance_attr, get_name, ensure_path, get_from_dict, search_dict
from utils_model import init_weight, cal_acc_from_label, get_loss_func, get_act_func, get_mask, get_ei_mask, get_cons_func, cat_dict

class RSLP(nn.Module):
    def __init__(self, dict_=None, load=False, f=None):
        super(RSLP, self).__init__()
        self.dict = dict_
        self.device = self.dict['device']
        
        #set_instance_attr(self, self.dict, exception=['N'])
        #selo = Neurons_LIF(dict_ = self.dict['N'], load=load)
        #self.output_num = self.output_num
        
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
            self.r_b = self.dict['r_b'] # recurrent bias
            if isinstance(self.i_b, torch.Tensor):
                self.register_parameter('i_b', self.i_b)            
            if self.dict['init_weight'] in ['nonzero']:
                self.h_init = self.dict['h_init'] # init hidden state
                self.register_parameter('h_init', self.h_init)
        else:
            self.i = torch.nn.Parameter(torch.zeros((self.dict['input_num'], self.dict['N_num']), device=self.device))
            
            self.dict['i'] = self.i
            
            if self.dict['bias']:
                self.i_b = torch.nn.Parameter(torch.zeros((self.dict['N_num']), device=self.device))            
            else:
                self.i_b = 0.0
            self.dict['b_0'] = self.i_b

            self.dict['r_b'] = search_dict(self.dict, ['r_b', 'bias'], default=True, write_default=False)
            if self.dict['r_b']:
                self.r_b = self.dict['r_b'] = torch.nn.Parameter(torch.zeros((self.dict['input_num']), device=self.device))
            else:
                self.r_b = self.dict['r_b'] = 0.0
            self.o = self.dict['o'] = nn.Parameter(torch.zeros((self.dict['N_num'], self.dict['output_num']), device=self.device, requires_grad=True))
            self.r = self.dict['r'] = nn.Parameter(torch.zeros((self.dict['N_num'], self.dict['N_num']), device=self.device, requires_grad=True))
            
            if self.dict.get('init_weight') is None:
                self.dict['init_weight'] = {
                    'r': ['input', 1.0],
                    'o': ['input', 1.0],
                    'i': ['input', 1.0],
                }

            init_weight(self.i, self.dict['init_weight']['i'])
            init_weight(self.r, self.dict['init_weight']['r'])
            init_weight(self.o, self.dict['init_weight']['o'])

        # set up basic attributes
        self.step_num = self.dict['step_num']
        self.N_num = self.dict['N_num']
        if self.dict['separate_ei']:
            self.time_const_e = self.dict['time_const_e']
            self.time_const_i = self.dict['time_const_i']
            self.act_func = self.get_act_func_ei()
            self.act_func_e = get_act_func(self.dict['act_func_e'])
            self.act_func_i = get_act_func(self.dict['act_func_i'])
            self.E_num = self.dict['E_num']
            self.I_num = self.dict['I_num']
            self.cal_s = self.cal_s_ei
            self.get_weight = self.get_weight_ei
            self.response = self.response_ei
            self.cache_weight = self.cache_weight_ei
            #self.weight_name = ['E->E','E->I','I->E','I->I','E->Y','I->Y','N->N','E.r','E.l','I.r','I.l','E.b','I.b','r','E.o','I.o','b']
            self.response_keys = ['E.u','E.x','I.u','I.x','E->E','E->I','I->E','I->I','E->Y','I->Y', 'X->E', 'X->I', 'N->Y', 'N->N', 'u']
        else:
            self.time_const = self.dict['time_const']
            self.act_func = get_act_func(self.dict['act_func'])
            self.cal_s = self.cal_s_uni
            self.get_weight = self.get_weight_uni
            self.response = self.response_uni
            self.cache_weight = self.cache_weight_uni
            #self.weight_name = ['X->E', 'X->I', 'i', 'N->Y','N->N', 'N.o', 'o', 'b', 'r']
            self.response_keys = ['o','r','u']

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



        # set up method to generate initial s, h
        self.init_mode = self.dict.setdefault('init_mode', 'zero')
        if self.init_mode in ['zero']:
            self.get_s_h_init = self.get_s_h_init_zero
        elif self.init_mode in ['learnable', 'fixed']:
            self.get_s_h_init = self.get_s_h_init_fixed
        else:
            raise Exception('Invalid s and h init mode: %s'%self.init_mode)

        # set up method to generate noise
        if self.dict['noise_coeff'] == 0.0:
            self.get_noise = lambda batch_size, N_num:0.0
        else:
            self.get_noise = self.get_noise_gaussian

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

        self.hebb_coeff = self.loss_dict.setdefault('hebb_coeff', 0.0)
        self.act_coeff = self.loss_dict.setdefault('act_coeff', 0.0)
        self.weight_coeff = self.loss_dict.setdefault('weight_coeff', 0.0)

        # performance log settings
        self.perform_list = {'class':0.0, 'act':0.0, 'weight':0.0, 'acc':0.0}
        if self.hebb_coeff != 0.0:
            self.perform_list['hebb'] = 0.0
        self.batch_count = 0
        self.sample_count = 0
        
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
    def forward_once(self, x):
        x = x.to(self.device)
        i = self.prep_input_once(x)
    def forward_once_no_grad(self, x):
        # to be implemented
        return
    def forward(self, x, step_num=None): # [batch_size, C x H x W]
        if step_num is None:
            step_num = self.step_num
        act_list = []
        output_list = []

        x = x.view(x.size(0), -1)
        
        self.prep_input(x)
        #self.reset_x(batch_size=x.size(0))
        #i_ = self.prep_input(x) # [step_num, batch_size, N_num]
        s = None
        h = None
        for time in range(step_num):
            state = self.forward_N(s=s, h=h, i=self.get_input(time))
            s, u, h, o = get_from_dict(state, ['s', 'u', 'h', 'o'])
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
    '''
    def reset(self, **kw):
        self.reset_x(**kw)

    def forward_once(self, x, h_in_detach, detach_i=True, detach_u=False, reset=False): # [batch_size, input_num]
        if reset:
            self.reset()
        #print(x.device) 
        x = x.view(x.size(0), -1)
        i = self.prep_input_once(x)
        if detach_i:
            i_ = i.detach()        
        else:
            i_ = i
        #print(i_.device)
        #print(h_in_detach.device)
        o, h_out, u = self.forward_once(i_ + h_in_detach, detach_u=detach_u)
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
    def forward_N(self, i):
        dx = i + self.b
        self.x = self.cal_s(dx) #x: [batch_size, N_num]
        u = self.act_func(self.x)
        o = u.mm(self.get_o())
        h = u.mm(self.get_r())
        return o, h, u
    '''
    def forward_N(self, s=None, h=None, i=None, detach_u=False):
        if s is None and h is None: # first iteration
            s, h = self.get_s_h_init(batch_size=i.size(0))   
        elif s is None or h is None:
            raise Exception('s and h must simultaneously be None or not None.')

        s = self.cal_s(h + self.r_b + i, s=s) #x: [batch_size, N_num]
        u = self.act_func(s)
        if detach_u:
            u_detach = u.detach()
            o = u_detach.mm(self.get_o())
            h = u_detach.mm(self.get_r())
            return {
                'u_detach': u_detach,
                's': s,
                'h': h,
                'o': o,
            }
        else:
            o = u.mm(self.get_o())
            h = u.mm(self.get_r())
            return {
                'u': u,
                's': s,
                'h': h,
                'o': o,
            }
    def forward_N_no_grad(self, s=None, h=None, i=None, detach_u=False):
        if s is None and h is None: # first iteration
            s, h = self.get_s_h_init(batch_size=i.size(0))
            s = s.detach()
            h = h.detach()    
        elif s is None or h is None:
            raise Exception('s and h must simultaneously be None or not None.')
        
        dx = i + self.r_b
        s = self.cal_s(h + self.r_b + i, s=s) #x: [batch_size, N_num]
        u = self.act_func(s)
        if detach_u:
            u_detach = u.detach()
            o = u_detach.mm(self.get_o().detach())
            h = u_detach.mm(self.get_r().detach())
            return {
                'u_detach': u_detach,
                's': s,
                'h': h,
                'o': o,
            }
        else:
            o = u.mm(self.get_o().detach())
            h = u.mm(self.get_r().detach())
            return {
                'u': u,
                's': s,
                'h': h,
                'o': o,
            }
    def get_s_h_init_zero(self, **kw):
        batch_size = kw['batch_size']
        return torch.zeros([batch_size, self.N_num], device=self.device), torch.zeros([batch_size, self.N_num], device=self.device)
    def get_s_h_init_fixed(self, **kw):
        batch_size = kw['batch_size']
        return torch.stack([self.s_init for _ in range(batch_size)], dim=0), torch.stack([self.h_init for _ in range(batch_size)], dim=0)
    '''
    def reset_x_zero(self, **kw):
        #print(batch_size)
        self.x = torch.zeros((kw['batch_size'], self.dict['N_num']), device=self.device) #(batch_size, input_num)
    '''
    def get_noise_gaussian(self, batch_size, N_num):
        noise = torch.zeros((batch_size, N_num), device=self.device)
        torch.nn.init.normal_(noise, 0.0, self.noise_coeff)
        return noise
    def get_act_func_ei(self):
        return lambda s: torch.cat( [self.act_func_e(s[:, 0:self.E_num]), self.act_func_i(s[:, self.E_num:self.N_num])], dim=1)
    def cal_s_uni(self, dx, s=None):
        if s is None:
            s = self.s
        return (1.0 - self.time_const) * (s + self.get_noise(dx.size(0), self.N_num)) + self.time_const * dx #x: [batch_size, N_num]
    def cal_s_ei(self, dx, s=None):
        if s is None:
            s = self.s
        s_e = (1.0 - self.time_const_e) * (s[:, 0:self.E_num] + self.get_noise(dx.size(0), self.E_num)) + self.time_const_e * dx[:, 0:self.E_num] # x: [batch_size, E_num]
        s_i = (1.0 - self.time_const_i) * (s[:, self.E_num:self.N_num] + self.get_noise(dx.size(0), self.I_num)) + self.time_const_i * dx[:, self.E_num:self.N_num] # x: [batch_size, I_num]        
        return torch.cat([s_e, s_i], dim=1)
    def response(self, x, step_num=None):
        if step_num is None:
            step_num = self.step_num
        x = x.view(x.size(0), -1)
        self.reset_x(batch_size=x.size(0))
        i_ = self.prep_input_full(x) #(batch_size, step_num, N_num)
        r = 0.0
        for time in range(step_num):
            i_tot = torch.squeeze(i_[:, time, :]) + r
            o, r, u, res = self.response(i_tot)
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
        self.reset_x(batch_size=x.size(0))
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
            o, r, u, res = self.response(i_tot)
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
        result = self.forward(x)
        output, act = get_from_dict(result, ['output', 'act'])
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

        loss_weight = self.weight_coeff * ( torch.mean(self.get_r() ** 2) )
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

        o_1 = self.get_o().detach().cpu().numpy()
        if self.cache.get('o') is not None:
            o_0 = self.cache['o']
            f_change_rate = np.sum(abs(o_1 - o_0)) / np.sum(np.abs(o_0))
            result += 'f_change_rate: %.3f '%f_change_rate
        self.cache['o'] = o_1

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
    '''
    def anal_weight_change(self):
        i = self.get_i().detach().cpu().numpy()
        r = self.get_r().detach().cpu().numpy()
        o = self.get_o().detach().cpu().numpy()

        i_delta = np.sum(np.abs(i - self.cache['i'])) / np.sum(np.abs(self.cache['i']))
        r_delta = np.sum(np.abs(r - self.cache['r'])) / np.sum(np.abs(self.cache['r']))
        f_delta = np.sum(np.abs(f - self.cache['o'])) / np.sum(np.abs(self.cache['o']))
        print('weight update rate: i:%.4e r:%.4e f:%.4e'%(i_delta, r_delta, f_delta))
    '''
    def cal_perform_hebb(self, act):
        x = torch.squeeze(act[-1, :, :]) # [batch_size, N_num]
        batch_size=x.size(1)
        x = x.detach().cpu().numpy()
        x = torch.from_numpy(x).to(self.device)
        weight=self.get_r() # [N_num, N_num]
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

    def update_weight_cache_ei(self):
        self.update_weight_cache_ei()
        self.cache['weight_cache'] = self.cache['weight_cache']
        #print('E->E: ')
        #print(self.cache['weight_cache']['E->E'])
        #print(self.get_weight('E->E'))
        #input()
    def get_iter_data(self, data, step_num=None, batch_num=None):
        print('calculating iter_data. batch_num=%d'%(len(data)))
        if(step_num is None):
            step_num = self.step_num
        self.update_weight_cache()
        self.eval()
        count = 0
        data = list(data)
        if batch_num is None:
            batch_num = len(data)
        else:
            data = random.sample(data, 50)
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
        for batch in data:
            inputs, labels = batch
            inputs = inputs.to(self.device)
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
        self.cache['o'] = self.get_o().detach().cpu().numpy()


    def response_uni(self, i_):
        res = {}
        dx = i_ + self.b
        self.x = self.cal_s(dx) #x: [batch_size, N_num]
        res['x'] = self.x
        u = self.act_func(self.x)
        res['u'] = u
        res['o'] = u.mm(self.get_o())
        res['r'] = u.mm(self.get_r())
        return res
    def response_ei(self, i_):
        res = {}
        dx = i_ + self.b
        self.x = self.cal_s(dx) #x: [batch_size, N_num]
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
        
        o = u.mm(self.get_o())
        r = u.mm(self.get_r())
        res['N->Y'] = o
        res['N->N'] = r

        res['E->Y'] = torch.mm(res['E.u'], self.cache['weight_cache']['E->Y'])
        res['I->Y'] = torch.mm(res['I.u'], self.cache['weight_cache']['I->Y'])
        return o, r, u, res
  
    def get_weight_ei(self, name, detach=True): # deprecated: positive
        #sig_r = oalse
        #sig_o = oalse
        # input weight
        if name in ['X->E']:
            w = self.get_i()[:, 0:self.E_num]
        elif name in ['X->I']:
            w = self.get_i()[:, self.E_num:self.N_num]
        # input bias
        elif name in ['b_0']:
            w = self.i_b
        elif name in ['i', 'input']:
            return self.get_weight_uni(name, detach=detach)
        
        # recurrent weight
        elif name in ['E.r', 'E->E']:
            w = self.get_r()[0:self.E_num, 0:self.E_num]
        elif name in ['I.r', 'I->I']:
            w = self.get_r()[self.E_num:self.N_num, self.E_num:self.N_num]
            #sig_r = True
        elif name in ['E.l', 'E->I']:
            w = self.get_r()[0:self.E_num, self.E_num:self.N_num]
        elif name in ['I.l', 'I->E']:
            w = self.get_r()[self.E_num:self.N_num, 0:self.E_num]
            #sig_r = True
        elif name in ['r', 'N->N']:
            return self.get_weight_uni(name, detach=detach)
        # recurrent bias
        elif name in ['E.b', 'r_bias_E']:
            if isinstance(self.b, torch.Tensor):
                w = self.b[0:self.E_num]
            else:
                return self.b
        elif name in ['I.b', 'r_bias_I']:
            if isinstance(self.b, torch.Tensor):
                w = self.b[self.E_num:self.N_num]
            else:
                return self.b
        elif name in ['r_b', 'r_bias', 'b', 'bias']:
            return self.get_weight_uni(name, detach=detach)
        # output weight
        elif name in ['E.o', 'E->Y']:
            w = self.get_o()[0:self.E_num, :]
        elif name in ['I.o', 'I->Y']:
            w = self.get_o()[self.E_num:self.N_num, :]
            #sig_o = True
        elif name in ['o', 'output', 'N->Y']:
            return self.get_weight_uni(name, detach=detach)
        else:
            raise Exception('Invalid weight name:%s'%name)
        if detach:
            w = w.detach()
        '''
        if positive and sig_r and 'r' in self.dict['Dale']:
            w = - w
        elif positive and sig_f and 'o' in self.dict['Dale']:
            w = - w
        '''
        return w
    def get_weight_uni(self, name, detach=True):
        if name in ['r', 'N->N']:
            return self.get_r()
        elif name in ['b']:
            return self.get_b()
        elif name in ['o', 'output', 'N->Y']:
            w = self.get_o()
        else:
            raise Exception('Invalid weight name: %s'%name)
        if detach and isinstance(w, torch.Tensor):
            w = w.detach()
        return w
    def cache_weight_ei(self):
        self.cache['weight'] = {}
        weight_cache = self.cache['weight']
        N_r = self.get_weight('r')
        weight_cache['E->E'] = N_r[0:self.E_num, 0:self.E_num]
        weight_cache['I->I'] = N_r[self.E_num:self.N_num, self.E_num:self.N_num]
        weight_cache['E->I'] = N_r[0:self.E_num, self.E_num:self.N_num]
        weight_cache['I->E'] = N_r[self.E_num:self.N_num, 0:self.E_num]
        N_o = self.get_weight('o')
        weight_cache['E->Y'] = N_f[0:self.E_num, :]
        weight_cache['I->Y'] = N_f[self.E_num:self.N_num, :]
    def cache_weight_uni(self):
        # to be implemented
        return