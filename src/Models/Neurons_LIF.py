import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_instance_attr
from utils_model import init_weight, get_ei_mask, get_mask, get_cons_func, get_act_func

#training parameters.
class Neurons_LIF(nn.Module):
    def __init__(self, dict_, load=False):#input_num is N_num.
        super(Neurons_LIF, self).__init__()
        self.dict = dict_
        
        set_instance_attr(self, self.dict)
        #self.device = self.dict['device']
        
        if load:
            self.o = self.dict['o']
            self.r = self.dict['r']
            self.b = self.dict['b']
            if self.dict['init_weight'] in ['nonzero']:
                self.init_state = self.dict['init_state']
        else:
            if self.dict['bias']:
                self.b = torch.nn.Parameter(torch.zeros((self.dict['input_num']), device=self.device))
            else:
                self.b = 0.0
            self.dict['b'] = self.b
            self.o = torch.nn.Parameter(torch.zeros((self.dict['N_num'], self.dict['output_num']), device=self.device, requires_grad=True))
            self.r = torch.nn.Parameter(torch.zeros((self.dict['N_num'], self.dict['N_num']), device=self.device, requires_grad=True))
            self.dict['o'] = self.o
            self.dict['r'] = self.r
            
            init_weight(self.r, self.dict['init_weight']['r'])
            init_weight(self.o, self.dict['init_weight']['o'])             

        # set recurrent weight
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

        # set forward weight
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

        if self.noise_coeff == 0.0:
            self.get_noise = lambda batch_size, N_num:0.0
        else:
            self.get_noise = self.get_noise_gaussian

        self.cache = {}
        self.reset_x = self.reset_x_zero
        self.x = None #to be set

        #print('r:')
        #print(self.dict['noself'])
        #print(self.get_r())
        #print(self.ei_mask)
        #print(self.get_weight(name='I->I'))
        #input()
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
    def forward(self, i):
        dx = i + self.b
        self.x = self.cal_x(dx) #x: [batch_size, N_num]
        u = self.act_func(self.x)
        o = u.mm(self.get_o())
        h = u.mm(self.get_r())
        return o, h, u
    def forward_once(self, x=None, h=None, i=None, detach_u=False):
        if x is None:
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
        if(name in ['r', 'N->N']):
            return self.get_r()
        elif(name=='b'):
            return self.get_b
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