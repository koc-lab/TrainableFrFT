# %%
import torch
import torchvision
from torch.utils.data.distributed import DistributedSampler

from configurations.configs import DataHandlerConfig

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset, Dataset, random_split

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

import pandas as pd
import numpy as np
import os


import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class CustomDataHandler:
    def __init__(
        self,
        config: DataHandlerConfig,
       
):     
        self.c = config
        self._make_loaders()
      
      

    def _make_loaders(
        self,
    ):
        self.trainset, self.testset = self.get_datas(self.c.dataset)

        self.trainloader = self.get_dataloader(self.trainset)
        self.testloader = self.get_dataloader(self.testset)
        self.loaders = dict(train=self.trainloader, test=self.testloader)

    def get_dataloader(self, dataset: torchvision.datasets):
        if self.c.multi_gpu:
            sampler = DistributedSampler(dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            dataset=dataset,
            batch_size=self.c.batch_size,
            pin_memory=True,
            shuffle=shuffle,
            sampler=sampler,
        )

    def get_datas(self, dataset_name):
       
        if dataset_name=="CIFAR-10":
            
            root="./data"
            train_dataset = CIFAR10(
                root=root, train=True, transform=self.c.train_transform, download=True
            )
            test_dataset = CIFAR10(root=root, train=False, transform=self.c.test_transform)

        elif dataset_name=="CUB2011":
            
            root="./data/cub2011_data"
            train_dataset = Cub2011(root=root, train=True, download=True, transform=self.c.train_transform)
            test_dataset=Cub2011(root=root, train=False, download=True, transform=self.c.test_transform)

        sub_train_dataset = Subset(
                train_dataset, indices=range(0, len(train_dataset), self.c.train_slice)
            )
        sub_test_dataset = Subset(
                test_dataset, indices=range(0, len(test_dataset), self.c.test_slice)
            )
        return sub_train_dataset, sub_test_dataset


import torchvision.transforms as transforms


def get_basic_transform(dataset_name):
    # TODO: only for cifar10 now, make it generic to all datasets
    
    if dataset_name=="CIFAR-10":
            d_mean = [0.49139968, 0.48215841, 0.44653091]
            d_std = [0.24703223, 0.24348513, 0.26158784]
            normalize = transforms.Normalize(d_mean, d_std)

            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

            transform_test = transforms.Compose([transforms.ToTensor(), normalize])
   
    elif dataset_name=="CUB2011":
            
            image_size=224

            transform_train = transforms.Compose([
                    
                    transforms.Resize(int(image_size/0.875)),
                    transforms.CenterCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(30),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225))
                ])
            transform_test = transforms.Compose([
                    transforms.Resize(int(image_size/0.875)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225))
                ])
        
   

    return transform_train, transform_test
##############################################################3

# Define CUB2011 DATASET
class Cub2011(Dataset):

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
        self.tgz_md5 = '97eceeb196236b17998738112f37df78'
        self.filename = 'CUB_200_2011.tgz'
        self.base_folder = 'CUB_200_2011/images/'
        #self._load_metadata()

        if download:
            self._download()

        if not self._check_integrity():
           raise RuntimeError('Dataset not found or corrupted.' +
                               'You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')


        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True


    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to
        img = self.loader(path)
        #convert_tensor = transforms.ToTensor()

        #img=convert_tensor(img)


        if self.transform is not None:
            img = self.transform(img)

        return img, target
###############################################################################################3
    
'''
 Load CUB2011_data
'''

def cub2011_data_loader(data_dir,batch_size=128,train_split=0.9):
   
    image_size=224


    train_transform = [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
    test_transform = [
            transforms.Resize(int(image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
  
    # Data
    print('==> Preparing data..')

    transform_train =transforms.Compose(train_transform)
    transform_test =transforms.Compose(test_transform)


    train_dataset = Cub2011(
        root=data_dir, train=True, download=True, transform=transform_train)
    
    
    # Calculate the sizes for training and validation sets
    num_samples = len(train_dataset)
    train_size = int(train_split * num_samples)
    val_size = num_samples - train_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoader instances for training and validation sets
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    testset = Cub2011(
        root=data_dir, train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size)#, num_workers=2)

    return trainloader,validloader,testloader



