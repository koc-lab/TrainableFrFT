# %%
import torchvision
from torch.utils.data.distributed import DistributedSampler

from configurations.configs import DataHandlerConfig

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset


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
        self.trainset, self.testset = self.get_datas()

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

    def get_datas(self, root: str = "./data"):
        train_dataset = CIFAR10(
            root=root, train=True, transform=self.c.train_transform, download=True
        )

        test_dataset = CIFAR10(root=root, train=False, transform=self.c.test_transform)

        sub_train_dataset = Subset(
            train_dataset, indices=range(0, len(train_dataset), self.c.train_slice)
        )

        sub_test_dataset = Subset(
            test_dataset, indices=range(0, len(test_dataset), self.c.test_slice)
        )

        return sub_train_dataset, sub_test_dataset


import torchvision.transforms as transforms


def get_basic_transform():
    # TODO: only for cifar10 now, make it generic to all datasets

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

    return transform_train, transform_test


# def cifar_data_loader(data_dir, batch_size=128, train_split=0.9):
#     # Previously calculated
#     d_mean = [0.49139968, 0.48215841, 0.44653091]
#     d_std = [0.24703223, 0.24348513, 0.26158784]
#     normalize = transforms.Normalize(d_mean, d_std)

#     # Data
#     print("==> Preparing data..")
#     transform_train = transforms.Compose(
#         [
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]
#     )

#     transform_test = transforms.Compose([transforms.ToTensor(), normalize])

#     train_dataset = CIFAR10(data_dir, transform=transform_train)

#     # Calculate the sizes for training and validation sets
#     num_samples = len(train_dataset)
#     train_size = int(train_split * num_samples)
#     val_size = num_samples - train_size

#     # Use random_split to create training and validation datasets
#     train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

#     # Create DataLoader instances for training and validation sets
#     trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     validloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     testset = CIFAR10(data_dir, train=False, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=batch_size
#     )  # , num_workers=2)

#     return trainloader, validloader, testloader
