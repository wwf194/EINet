import config_ML

dict_ = {
    'data_set': 'cifar10',
    'num_class': 10,
    'data_path': config_ML.dir_cifar10,
    'gray': False # whether to make image grey.
}

if dict_.get('type') is None:
    dict_['type'] = dict_['data_set']


if dict_.get('data_dir') is None:
    dict_['data_dir'] = dict_['data_path']

def interact(env_info):
    trainer_dict = env_info['trainer_dict']
    if trainer_dict is not None:
        #print(trainer_dict.keys())
        if trainer_dict.get('batch_size') is not None:
            dict_['batch_size'] = trainer_dict['batch_size']
    



