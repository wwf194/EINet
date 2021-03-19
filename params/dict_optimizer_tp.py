lr = 1.0e-2
lr_decoder = 10 * lr
sub_optimizer_type = 'sgd'
dict_ ={
    'class': 'optimizer',
    'loss_main': 'MSE',
    'type': 'tp',
    'lr': lr,
    'sub_optimizer_type': sub_optimizer_type, # optimizer used to train encoder and decoder weights.
    
    'lr_loss': 'MSE',
    'lr_decoder': lr_decoder,
    'decoder_rec':{
        'type':'mlp',
        'act_func':'relu',
        'bias': True,
        'lr': lr_decoder
    },
    'decoder_out':{
        'type':'mlp',
        'act_func':'relu',
        'bias': True,
        'lr': lr_decoder
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
        'milestones': [[0.50, 1.0],[0.70, 1.0e-1], [0.85, 1.0e-2], [0.95, 1.0e-3]],
    },
    'mode':'train_on_h', # train_on_h or train_on_u
    'get_target_method': 'gradient',
}

def interact(env_info):
    return

def interact(env_info):
    model_dict = env_info['model_dict']
    if env_info.get('device') is not None:
        dict_['device'] = env_info['device']
    if model_dict is not None:
        dict_['loss_func'] = model_dict['loss']['main_loss_func']
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