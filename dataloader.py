# %%
import torch
import torchvision
import torchvision.transforms as transforms


import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

from vgg_variants import CLASS_NAMES


def get_data(
    transform: torchvision.transforms,
    slice: int = 1,
    train: bool = True,
    root: str = "./data",
):
    """
    slice: selecting elements from the dataset with for loop context
    """

    # TODO: add a check for the slice value

    full_dataset = CIFAR10(root=root, train=train, transform=transform, download=True)

    sub_dataset = Subset(full_dataset, indices=range(0, len(full_dataset), slice))

    return sub_dataset


def get_loader(
    dataset: torchvision.datasets,
    batch_size: int,
):
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return loader


def imshow(img: torch.Tensor):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def make_loaders(
    batch_size,
    train_slice=1,
    test_slice=1,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ),
):
    trainset = get_data(transform=transform, slice=train_slice, train=True)
    testset = get_data(transform=transform, slice=test_slice, train=False)

    trainloader = get_loader(trainset, batch_size=batch_size)
    testloader = get_loader(testset, batch_size=batch_size)

    return dict(train=trainloader, test=testloader)


# %%


def main():
    batch_size = 4

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = get_data(transform=transform, slice=1, train=True)
    trainloader = get_loader(trainset, batch_size=batch_size)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print(" ".join(f"{CLASS_NAMES[labels[j]]:5s}" for j in range(batch_size)))

    for i, (images, labels) in enumerate(trainloader):
        if i == 0:
            print(images.shape)
            imshow(torchvision.utils.make_grid(images))
            print(labels)

            break


if __name__ == "__main__":
    main()

# %%
