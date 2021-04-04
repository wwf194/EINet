lr = 1.0e-2
lr_decoder = 10 * lr
sub_optimizer_type = 'sgd'
decoder_type = 'mlp'
decoder_act_func = 'relu'
decoder_bias= True
decoder_batch_norm = True
dict_ ={
    'class': 'optimizer',
    'loss':{
        'main_loss': 'MSE', # loss function to compute main loss
        'main_loss_coeff': 1.0,
        'act_coeff': 0.0e-4, # coefficient of L2 loss of average fire rate over time_step and neuron
    },
    'decoder_loss': 'MSE',
    'type': 'tp',
    'lr': lr,
    'sub_optimizer_type': sub_optimizer_type, # optimizer used to train encoder and decoder weights.
    'lr_decoder': lr_decoder,
    'decoder_rec':{
        'type': decoder_type,
        'act_func': decoder_act_func,
        'bias': decoder_bias,
        'lr': lr_decoder,
        #'alter_weight_scale': True, # a simple way to alter weight scale to match output level.
        'batch_norm': decoder_batch_norm,
    },
    'decoder_out':{
        'type': decoder_type,
        'act_func': decoder_act_func,
        'bias': decoder_bias,
        'lr': lr_decoder,
        #'alter_weight_scale': True,
        'batch_norm': decoder_batch_norm, # batch norm automatically alterate scale of weights and biases are to appropriate scale.
    },
    'optimizer_forward':{
        'lr':lr,
        'type':sub_optimizer_type,
    },
    'optimizer_rec':{
        'lr':lr_decoder,
        'type':sub_optimizer_type,
    },
    'optimizer_out':{
        'lr':lr_decoder,
        'type':sub_optimizer_type,
    },
    'lr_decay':{
        'method': 'linear',
        'milestones': [[0.50, 1.0], [0.70, 1.0e-1], [0.85, 1.0e-2], [0.95, 1.0e-3]],
    },
    'mode':'train_on_h', # train_on_h or train_on_u
    'get_target_method': 'gradient',
}

def interact(env_info):
    model_dict = env_info['model_dict']
    if env_info.get('device') is not None:
        dict_['device'] = env_info['device']
    #if model_dict is not None:
    #    dict_['loss_func'] = model_dict['loss']['main_loss']
    if env_info.get('data_loader_dict') is not None:
        if env_info['data_loader_dict'].get('num_class') is not None:
            dict_['loss']['num_class'] = env_info['data_loader_dict']['num_class']
    return

'''
set_default_attr(param, 'lr_decay_method', 'None')
if lr_decay_method in ['linear', 'Linear']:
    dict_optimizer['lr_decay'] = {
        'method': 'linear',
        'milestones': milestones,
    }
    if kw.get('dict_trainer') is not None:
        dict_optimizer['lr_decay']['epoch_num'] = kw.get('dict_trainer')['epoch_num']
elif lr_decay_method in ['log']:
    dict_optimizer['lr_decay'] = { #learning rate decay
        'method':'log',
        'milestones':[[1.0, 1.0e-3]],
    }
elif lr_decay_method in ['None']:
    dict_optimizer['lr_decay'] = {'method':'None'}
else:
    raise Exception('def_optimizer_param: Invalid lr_decay_method:%s'%(str(lr_decay_method)))
'''

'''
'bias': True,
'N_nums': [2, encoder_output_num, encoder_output_num],
'act_func': 'tanh',
'act_func_on_last_layer': False,
'''