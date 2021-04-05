N_num = 500
E_ratio = 0.8
E_num = int( N_num * E_ratio )
cons_method = 'abs'
noise_coeff = 0.0e-4 # standard devication of N(0, 1) Gaussian noise added to fire rate.

separate_ei = True
Dale = ['r', 'f']

init_weight = {
    'i':['input', 1.0e-3],
    'o':['input', 1.0e-3],
    'r':['input', 1.0e-3],
}

# dict of sinle-layer perceptron
dict_ = {
    'type': 'RSLP',
    'name': 'RSLP',
    #'N_num': N_num,
    'step_num': 10,
    #'init_weight': init_weight,
    #'noise_coeff': noise_coeff, 
    'separate_ei': separate_ei,
    'E_ratio': 0.8, # valid only if separate_ei is True
    'bias': False,
    'cons_method': cons_method,
    'output_num': None,
    #'N':{
    'N_num': N_num,
    'E_num': E_num,
    'I_num': N_num - E_num,
    'output_num': None,
    'init_weight': init_weight,
    'bias': False,
    'noself': True,
    'mask': [],
    'separate_ei': separate_ei,
    'cons_method': cons_method,
    'noise_coeff': noise_coeff,
    'Dale': Dale,
    #},
    'input_mode': 'endure',
}

if separate_ei:
    dict_.update({
        'Dale': Dale,
    })
    dict_.update({
        'Dale': Dale, # which weight conforms to Dale's law. r for recurrent weight, f for output(feedforward) weight.
        'act_func_e':['relu', 1.0],
        'act_func_i':['relu', 1.0],
        'time_const_e': 0.4, 
        'time_const_i': 0.2,
    })
else:
    dict_.update({
        'act_func':['relu', 1.0],
        'time_const': 0.2,
    })

if dict_.get('iter_time') is None:
    dict_['iter_time'] = dict_['step_num']

if dict_['separate_ei']:
    dict_['name'] = 'RSLP(EI)'
else:
    dict_['name'] = 'RSLP(no_EI)'

def interact(env_info):
    data_loader_dict = env_info['data_loader_dict']

    if data_loader_dict is not None:
        data_loader_type = data_loader_dict['type']
        if data_loader_type in ['cifar10']:
            if data_loader_dict['gray']:
                dict_['input_num'] = 32 * 32
                #dict_['N']['output_num'] = 10
                dict_['output_num'] = 10
            else:
                dict_['input_num'] = 32 * 32 * 3
                #dict_['output_num'] = dict_['N']['output_num'] = 10
                dict_['output_num'] = 10
        elif data_loader_type in ['mnist']:
            dict_['input_num'] = 28 * 28
            #dict_['output_num'] = dict_['N']['output_num'] = 10
            dict_['output_num'] = 10
        else:
            raise Exception('Unknown data loader type: %s'%data_loader_type)

    if env_info.get('device') is not None:
        #dict_['device'] = dict_['N']['device'] = env_info['device']
        dict_['device'] = env_info['device']
    if env_info.get('optimizer_dict') is not None:
        dict_['loss'] = env_info['optimizer_dict']['loss']
