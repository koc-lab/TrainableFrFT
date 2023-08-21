import torch
import torch.nn as nn
from models.custom_frft_layers import FrFTPool, DFrFTPool, FFTPool

from typing import Union

from configurations.configs import PoolType
from utils.utils import get_model_variants_dict

VGG_LAYER_DICT = get_model_variants_dict("vgg")


class VGG(nn.Module):
    def __init__(self, model_name: str, n_class):
        super(VGG, self).__init__()
        self.model_name = model_name
        self.n_class = n_class

        self.vgg_block = VGGBlock(VGG_LAYER_DICT[model_name])
        self.classifier = nn.Linear(512, n_class)

    def forward(self, x):
        out = self.vgg_block(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    # TODO: add a method to get the fractional orders of the model
    def get_frac_orders(self):
        d = {}
        pool_count = 1
        for layer in self.vgg_block.layers:
            if isinstance(layer, FrFTPool) or isinstance(layer, DFrFTPool):
                order_1, order_2 = layer.order1.item(), layer.order2.item()

                # TODO: add modulo with tuna
                if isinstance(layer, FrFTPool):
                    d[f"pool_{pool_count}"] = (order_1 % 4, order_2 % 4)
                elif isinstance(layer, DFrFTPool):
                    d[f"pool_{pool_count}"] = (order_1 % 4, order_2 % 4)

            pool_count += 1

        return d


# ?: Since layer config is predefined and we do not get input from user, I define it as a class attribute
# ?: instead of a class
# ?: I use layer config as class attribute which allows to use it in the forward function, is that meaningful?


class VGGBlock(nn.Module):
    def __init__(self, layer_config: list[Union[int, PoolType]]):
        super(VGGBlock, self).__init__()

        self.layer_config = layer_config
        self.layers = self._make_vgg_block()

    def _make_vgg_block(self):
        in_channel, layers = 3, []

        for val in self.layer_config:
            if val is PoolType.MaxPool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif val is PoolType.FrFTPool:
                layers.append(FrFTPool())

            elif val is PoolType.FFTPool:
                layers.append(FFTPool())

            elif val is PoolType.DFrFTPool:
                layers.append(DFrFTPool())

            elif isinstance(val, int):
                layers.append(
                    ConvBlock(
                        in_channel=in_channel, out_channel=val, kernel_size=3, padding=1
                    )
                )
                in_channel = val

            else:
                raise ValueError(
                    f"Invalid layer type {val} in VGG. Must be either int or PoolType"
                )

        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# test()

# %%
