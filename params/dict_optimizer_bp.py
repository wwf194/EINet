lr = 1.0e-2

dict_ ={
    'class': 'optimizer',
    'type': 'bp',
    'loss':{
        'main_loss': 'MSE', # loss function to compute main loss
        'main_loss_coeff': 1.0,
        'act_coeff': 0.0e-4, # coefficient of L2 loss of average fire rate over time_step and neuron
    },
    'optimizer_dict':{
        'type': 'sgd',
        'lr': lr
    },
    'lr_decay': {
        'method': 'linear',
        'milestones': [[0.50, 1.0],[0.70, 1.0e-1], [0.85, 1.0e-2], [0.95, 1.0e-3]],
    }
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