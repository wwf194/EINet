# run this script to do different tasks

# python main.py --task copy --path ./Instances/TP-1/  //move necessary files for training and analysis to path.
# python main.py -t backup -p /data4/wangweifan/.backup/Hebb //backup project folder files to another given path.

# Every component, such as model, optimizer, trainer, data_loader, is initialized according to a dict in a .py file.

import os
import sys
import re
import argparse
import warnings
import importlib
import shutil

sys.path.append('./src/')

import config_sys
from utils import build_model, build_optimizer, build_trainer, build_data_loader, get_device, remove_suffix, select_file, ensure_path
from utils import scan_files, copy_files, path_to_module, copy_folder
from Trainers import Trainer
import Models
import Optimizers

parser = argparse.ArgumentParser(description='Parse args.')
parser.add_argument('-d', '--device', type=str, dest='device', default=None, help='device')
parser.add_argument('-t', '--task', type=str, dest='task', default=None, help='task to do')
parser.add_argument('-p', '--path', type=str, dest='path', default=None, help='a path to current directory. required in some tasks.')
parser.add_argument('-o', '--optimizer', dest='optimizer', type=str, default=None, help='optimizer type. BP, TP, CHL, etc.')
parser.add_argument('-tr', '--trainer', dest='trainer', type=str, default=None, help='trainer type.')
parser.add_argument('-m', '--model', dest='model', type=str, default=None, help='model type. RSLP, RMLP, RSLCNN, RMLCNN, etc.')
parser.add_argument('-dl', '--data_loader', dest='data_loader', type=str, default=None, help='data loader type.')
parser.add_argument('-pp', '--param_path', dest='param_path', type=str, default=None, help='path to folder that stores param dict files.')
parser.add_argument('-cf', '--config', dest='config', type=str, default=None, help='name of config file')
args = parser.parse_args()

def scan_param_files(path):
    if not path.endswith('/'):
        path.append('/')
    model_files = scan_files(path, r'dict_model(.*)\.py', raise_not_found_error=False)
    optimizer_files = scan_files(path, r'dict_optimizer(.*)\.py', raise_not_found_error=False)
    trainer_files = scan_files(path, r'dict_trainer(.*)\.py', raise_not_found_error=False)
    data_loader_files = scan_files(path, r'dict_data_loader(.*)\.py', raise_not_found_error=False)
    config_files = scan_files(path, r'config(.*)\.py', raise_not_found_error=False)

    '''
    if raise_not_found_error: # raise error if did not find any param dict
        if len(model_files)==0:
            raise Exception('No available model param dict in %s'%str(path))
        if len(optimizer_files)==0:
            raise Exception('No available optimizer param dict in %s'%str(path))
        if len(trainer_files)==0:
            raise Exception('No available trainer param dict in %s'%str(path))
        if len(data_loader_files)==0:
            raise Exception('No available data_loader param dict in %s'%str(path)) 
    '''
    return {
        'model_files': model_files,
        'optimizer_files': optimizer_files,
        'trainer_files': trainer_files,
        'data_loader_files': data_loader_files,
        'config_files': config_files
    }
    '''
    files_path = os.listdir(path)
    pattern_model = re.compile(r'dict_model(.*)\.py')
    pattern_optimizer = re.compile(r'dict_optimizer(.*)\.py')
    pattern_trainer = re.compile(r'dict_trainer(.*)\.py')
    patern_data_loader = re.compile(r'dict_data_loader(.*)\.py')
    model_files, optimizer_files, trainer_files, data_loader_files = [], [], [], []
    for file_name in files_path:
        #print(file_name)
        if pattern_model.match(file_name) is not None:
            model_files.append(file_name)
        elif pattern_optimizer.match(file_name) is not None:
            optimizer_files.append(file_name)
        elif pattern_trainer.match(file_name) is not None:
            trainer_files.append(file_name)
        elif patern_data_loader.match(file_name) is not None:
            data_loader_files.append(file_name)
        else:
            #warnings.warn('Unidentifiable param dict: %s'%str(file_name))
            pass

    # remove folders
    for files in [model_files, optimizer_files, trainer_files, data_loader_files]:
        for file in files:
            if os.path.isdir(file):
                warnings.warn('%s is a folder, and will be ignored.'%(path + file))
                files.remove(file)

    return model_files, optimizer_files, trainer_files, data_loader_files
    '''

def get_param_files(args, verbose=True):
    path = args.param_path
    if path is None:
        path = './params/'
    if not path.endswith('/'):
        path += '/'
    files = scan_param_files(path)
    
    model_files = files['model_files']
    optimizer_files = files['optimizer_files']
    trainer_files = files['trainer_files']
    data_loader_files = files['data_loader_files']
    config_files = files['config_files']

    #print(model_files)
    #print(optimizer_files)
    #print(trainer_files)
    #print(data_loader_files)

    model_str, optimizer_str, trainer_str, data_loader_str = args.model, args.optimizer, args.trainer, args.data_loader
    
    files_str = [model_str, optimizer_str, trainer_str, data_loader_str]
    component_files = [model_files, optimizer_files, trainer_files, data_loader_files]
    
    use_config_file = False
    if len(config_files)==1:
        sig = True
        for files in component_files:
            if len(files)>1 or len(files)==0:
                sig = False
        if sig:
            use_config_file = True
    if args.config is not None:
        use_config_file = True

    if use_config_file: # get param files according to a config file.
        if verbose:
            print('Setting params according to config file.')
        if len(config_files)==1:
            config_file = config_files[0]
        else:
            config_file = select_file(args.config, config_files, default_file=None, 
                match_prefix='config_', match_suffix='.py', file_type='config')
        print(config_file)
        Config_Param = importlib.import_module(path_to_module(path) + remove_suffix(config_file))
        try:
            model_file = Config_Param.model_file
            optimizer_file = Config_Param.optimizer_file
            trainer_file = Config_Param.trainer_file
            data_loader_file = Config_Param.data_loader_file
        except Exception:
            raise Exception('Cannot read file name from %s'%(path + config_file))
    
        for file in [model_file, optimizer_file, trainer_file, data_loader_file]:
            if not os.path.exists(path + file):
                raise Exception('FileNotFoundError: %s'%(path + file))
        return {
            'model_file': model_file,
            'optimizer_file': optimizer_file,
            'trainer_file': trainer_file,
            'data_loader_file': data_loader_file,
            'config_file': config_file,
            'files_path': path,
        }
    else:
        if verbose:
            print('Setting params according to model, optimzier, trainer, and data_loader param files.')
        if len(model_files)==0:
            raise Exception('No available model param file.')
        elif len(model_files)==1:
            model_file = model_files[0]
            if verbose:
                print('Using the only available model file: %s'%model_file)          
        else:
            model_file = select_file(model_str, model_files, default_file='dict_model_rslp.py', 
                match_prefix='dict_model_', match_suffix='.py', file_type='model')
        
        if len(optimizer_files)==0:
            raise Exception('No available optimizer param file.')
        elif len(optimizer_files)==1:
            optimizer_file = optimizer_files[0]
            if verbose:
                print('Using the only available optimizer file: %s'%optimizer_file)        
        else:
            optimizer_file = select_file(optimizer_str, optimizer_files, default_file='dict_optimizer_bp.py', 
                match_prefix='dict_optimizer_', match_suffix='.py', file_type='optimizer')

        if len(trainer_files)==0:
            raise Exception('No available trainer param file.')
        elif len(trainer_files)==1:
            trainer_file = trainer_files[0]
            if verbose:
                print('Using the only available trainer file: %s'%trainer_file)
        else:
            trainer_file = select_file(trainer_str, trainer_files, default_file='dict_trainer.py', 
                match_prefix='dict_trainer_', match_suffix='.py', file_type='trainer')

        if len(data_loader_files)==0:
            raise Exception('No available data_loader param file.')
        elif len(data_loader_files)==1:
            data_loader_file = data_loader_files[0]
            if verbose:
                print('Using the only available data_loader file: %s'%data_loader_file)
        else:
            data_loader_file = select_file(data_loader_str, data_loader_files, default_file='dict_data_loader_cifar10.py', 
                match_prefix='dict_data_loader_', match_suffix='.py', file_type='data loader')
        
        #print(model_file)
        #print(optimizer_file)
        #print(trainer_file)
        #print(data_loader_file)
        
        return {
            'model_file': model_file,
            'optimizer_file': optimizer_file,
            'trainer_file': trainer_file,
            'data_loader_file': data_loader_file,
            'files_path': path,
        }
def get_param_dicts(args):
    component_files = get_param_files(args)
    model_file = component_files['model_file']
    optimizer_file = component_files['optimizer_file']
    trainer_file = component_files['trainer_file']
    data_loader_file = component_files['data_loader_file']
    files_path = component_files['files_path']
    
    #print(path_to_module(files_path) + remove_suffix(model_file))
    Model_Param = importlib.import_module(path_to_module(files_path) + remove_suffix(model_file))
    model_dict = Model_Param.dict_

    Optimizer_Param = importlib.import_module(path_to_module(files_path) + remove_suffix(optimizer_file))
    optimizer_dict = Optimizer_Param.dict_

    Trainer_Param = importlib.import_module(path_to_module(files_path) + remove_suffix(trainer_file))
    trainer_dict = Trainer_Param.dict_

    Data_Loader_Param = importlib.import_module(path_to_module(files_path) + remove_suffix(data_loader_file))
    data_loader_dict = Data_Loader_Param.dict_

    device = get_device(args)
    print('Using device: %s'%str(device))

    env_info = {
        'model_dict': model_dict,
        'optimizer_dict': optimizer_dict,
        'trainer_dict': trainer_dict,
        'data_loader_dict': data_loader_dict,
        'device': device
    }

    Model_Param.interact(env_info)
    Optimizer_Param.interact(env_info)
    Trainer_Param.interact(env_info)
    Data_Loader_Param.interact(env_info)

    return model_dict, optimizer_dict, trainer_dict, data_loader_dict

def train(args=None, param_path=None, **kw):
    if args is None:
        args = kw.get('args')
    
    if param_path is None:
        if args.param_path is not None:
            param_path = args.param_path
        else:
            param_path = './params/'
        #sys.path.append(param_path)

    model_dict, optimizer_dict, trainer_dict, data_loader_dict = get_param_dicts(args)

    trainer = build_trainer(trainer_dict)
    data_loader = build_data_loader(data_loader_dict)
    trainer.bind_data_loader(data_loader)
    # model can be RSLP, RMLP, RCNN ...
    model = build_model(model_dict)
    # optimizer can be BP, TP or CHL optimizer.
    optimizer = build_optimizer(optimizer_dict)
    optimizer.bind_model(model)
    optimizer.bind_trainer(trainer)
    trainer.bind_model(model)
    trainer.bind_optimizer(optimizer)
    
    trainer.train() # the model needs some data from data_loader to get response properties.
    model.analyze(data_loader=data_loader)

def copy_project_files(args):
    path = args.path
    if path is None:
        raise Exception('copy_project_files: args.path must not be none. please give path to copy files to')
    ensure_path(args.path)
    if args.param_path is None:
        param_path = './params/'
    print(path)
    if not path.endswith('/'):
        path += '/'
    file_list = [
        #'cmd.py',
        'Models',
        'Optimizers',
        'Trainers.py',
        'DataLoaders.py',
        #'Analyzer.py',
        'utils.py',
        'utils_anal.py',
        'utils_model.py',
        'config_sys.py',
    ]
    copy_files(file_list, path_from='./src/', path_to=path + 'src/')
    file_list = [
        'main.py',
        'params/__init__.py'
    ]
    copy_files(file_list, path_from='./', path_to=path)
    param_files = get_param_files(args)
    #param_files = list(map(lambda file:param_path + file, param_files))
    model_file = param_files['model_file']
    optimizer_file = param_files['optimizer_file']
    trainer_file = param_files['trainer_file']
    data_loader_file = param_files['data_loader_file']
    component_files = [model_file, optimizer_file, trainer_file, data_loader_file]
    if param_files.get('config_file') is not None:
        component_files.append(param_files['config_file'])
    #print(component_files)
    copy_files(component_files, path_from=param_path, path_to=path + param_path)

def backup(args):
    if args.path is None:
        path_to = '/data4/wangweifan/.ssh/Hebb/'
    else:
        path_to = args.path
    copy_folder(path_from=os.path.abspath('../../'), path_to=path_to)

def test_mlp():
    from utils_model import build_mlp
    mlp_dict = {
        'bias': True,
        'N_nums': [2, 10, 100],
        'act_func': 'tanh',
        'act_func_on_last_layer': False,     
    }
    mlp = build_mlp(mlp_dict)

    

if __name__=='__main__':
    if args.task is None:
        task = 'train'
        warnings.warn('Task is not given from args. Using default task: train.')
    else:
        task = args.task
    if task in ['copy', 'copy_files', 'copy_file']: # copy necessary files for training and 
        copy_project_files(args)
    elif task in ['train']:
        train(args)
    elif task in ['backup']:
        backup(args)
    elif task in ['test_mlp']:
        test_mlp()
    else:
        raise Exception('Invalid task: %s'%str(task))