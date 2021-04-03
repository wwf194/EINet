import torch
import torchvision
import torchvision.transforms as transforms

from utils import get_name, get_args, ensure_path

def get_operation(op_str, DataLoader):
    #print(op_str)
    if op_str in ['toTensor', 'to_tensor']:
        #print('a')
        operation = transforms.ToTensor()
    elif op_str in ['norm','Norm','normalize','Normalize']:
        if isinstance(DataLoader, DataLoader_mnist):
            #print('b')
            operation = transforms.Normalize((0.1307), (0.3081))   
        elif isinstance(DataLoader, DataLoader_cifar10):
            #print('c')
            operation = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        else:
            #print('d')
            raise Exception('unknown Dataloader type: %s'%(DataLoader.__class__))
    else:
        #print('e')
        raise Exception('unknown operation description string: %s'%op_str)
    return operation

class DataLoader_mnist:
    def __init__(self, dict_, options=None):
        self.options = options
        self.dict = dict_
        if self.dict.get('pipeline') is None or self.dict.get('pipeline') in ['standard', 'default']: #use default pipeline. operand is, in has higher priority over and, or, not.
            self.dict['pipeline'] = {[
                'toTensor',
                'norm',
                ]}
        
        if self.dict.get('data_type') is None:
            self.dict['data_type'] = ['train', 'test']

        if self.dict.get('num_workers') is None:
            self.dict['num_workers'] = 0

        self.data_path = self.dict['data_path']
        self.num_class = self.dict['num_class'] = 10
        self.get_loader = self.get_data
    def get_loader(self):
        if self.dict.get('separate_train_test') is None:
            trans = []
            for operation_str in self.dict['pipeline']:
                operaiton = get_operation(operaiton_str, self)
                trans.append(operation)
            trans_train = transforms.Compose(trans_train)
            trans_test = transforms.Compose(trans_test)
        else: #separate pipeline for trainLoader and testLoader
            trans_train=[]
            trans_test=[]
            # to be implemented

        train_loader, test_loader = None, None

        if train is None and 'train' in self.dict['data_type']  or  train==True:
            train_set = torchvision.datasets.MNIST(root=self.data_path, train=True, transform=trans_train, download=False)
            train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=self.dict['num_workers'])
        
        if test is None and 'test' in self.dict['data_type']  or  test==True:
            test_set = torchvision.datasets.MNIST(root=self.data_path, train=False, transform=trans_test, download=False)
            test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=self.dict['num_workers'])

        return train_loader, test_loader

class DataLoader_cifar10:
    def __init__(self, dict_, options=None, load=False):
        self.options = options
        self.dict = dict_
        if self.dict.get('pipeline') is None or self.dict.get('pipeline') in ['standard', 'default']: #use default pipeline. operand is, in has higher priority over and, or, not.
            self.dict['pipeline'] = [
                'toTensor',
                'norm',
            ]
        
        if self.dict.get('data_type') is None:
            self.dict['data_type'] = ['train', 'test']

        if self.dict.get('num_workers') is None:
            self.dict['num_workers'] = 0

        self.data_path = self.dict['data_path']
        self.num_class = self.dict['num_class'] = 10
        self.get_loader = self.get_data

    def code_reference(self):
        # TenCrop augmentation
        if augment:
            feature_map_width=24
        else:
            feature_map_width=32
        TenCrop=[
            transforms.TenCrop(24),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(crop) for crop in crops]))
            ]
        trans_train.append(TenCrop)
        trans_test.append(TenCrop)

        trans_train.append(transforms.ToTensor())
        trans_test.append(transforms.ToTensor())

        # Other augmentations
        transforms.RandomCrop(24),
        transforms.RandomHorizontalFlip(),

    def get_data(self, train=None, test=None):
        if self.dict.get('separate_train_test') is None:
            trans = []
            for op_str in self.dict['pipeline']: #op_str: operation string
                operation = get_operation(op_str, self)
                trans.append(operation)
            trans_train = transforms.Compose(trans)
            trans_test = transforms.Compose(trans)
        else: #separate pipeline for trainLoader and testLoader
            trans_train=[]
            trans_test=[]
            # to be implemented

        train_loader, test_loader = None, None
        if train is None and 'train' in self.dict['data_type']  or  train==True:
            train_set = torchvision.datasets.CIFAR10(root=self.data_path, train=True, transform=trans_train, download=False)
            train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.dict['batch_size'], shuffle=True, num_workers=self.dict['num_workers'])
        
        if test is None and 'test' in self.dict['data_type']  or  test==True:
            test_set = torchvision.datasets.CIFAR10(root=self.data_path, train=False, transform=trans_test, download=False)
            test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.dict['batch_size'], shuffle=True, num_workers=self.dict['num_workers'])

        return train_loader, test_loader
    
    def save(self, save_path='./', save_name='./data_loader_cifar10'):
        ensure_path(save_path)
        with open(save_path + save_name, 'wb') as f:
            torch.save(self.dict, f)

'''
def prep_fashion(): # fashion_mnist  
    transform = transforms.Compose(
    [transforms.ToTensor()])

    trainset = torchvision.datasets.FashionMNIST(root='./data/fashion_MNIST', transform=transform, train=True, download=True)
    testset = torchvision.datasets.FashionMNIST(root='./data/fashion_MNIST', transform=transform, train=False, download=True)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=16)
    testloader = DataLoader(dataset=testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

    return trainloader, testloader
'''
