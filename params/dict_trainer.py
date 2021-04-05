dict_ = {
    'epoch_num': 100,
    'batch_size': 128,
    # params about saving model.
    'save_model':True,
    'save_before_train': True,
    'save_after_train': True,
    'save_model_interval': None,
    'save_model_path': './saved_models/',
    'anal_path': './anal/',
}

if dict_.get('save_model_interval') is None:
    dict_['save_model_interval'] = int(dict_['epoch_num']/10)

def interact(model_dict=None, trainer_dict=None, optimizer_dict=None, data_loader_dict=None, **kw):
    return