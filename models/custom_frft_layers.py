# %%
from torch_frft.dfrft_module import dfrft
from torch_frft.frft_module import frft

import torch
import torch.nn as nn


from torch_frft.dfrft_module import dfrft
from torch_frft.frft_module import frft
from typing import Callable, Optional, Union

# transform: Callable[
#             [torch.Tensor, Optional[Union[float, torch.Tensor]]], torch.Tensor
#         ],
# TODO: ensure type checking for FFT Pool
from torch_frft.dfrft_module import dfrft
from torch_frft.frft_module import frft


class FFTPool(nn.Module):
    def __init__(self) -> None:
        super(FFTPool, self).__init__()

    def forward(self, x):
        #
        out = torch.fft.fft2(x, norm="ortho")

        *_, H, W = out.size()

        st_H = H // 4
        end_H = H - st_H

        st_W = W // 4
        end_W = W - st_W

        out = out[..., st_H:end_H, st_W:end_W]
        out = torch.fft.ifft2(out, norm="ortho")

        return torch.abs(out)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class DFrFTPool(nn.Module):
    def __init__(self, order: float = 1) -> None:
        super(DFrFTPool, self).__init__()

        self.order1 = nn.Parameter(torch.tensor(order, dtype=torch.float32))
        self.order2 = nn.Parameter(torch.tensor(order, dtype=torch.float32))

    def forward(self, x):
        #
        out = dfrft(x, self.order1, dim=-1)
        out = dfrft(out, self.order2, dim=-2)

        *_, H, W = out.size()

        st_H = H // 4
        end_H = H - st_H

        st_W = W // 4
        end_W = W - st_W

        out = out[..., st_H:end_H, st_W:end_W]

        #
        out = dfrft(out, -self.order1, dim=-1)
        out = dfrft(out, -self.order2, dim=-2)

        return torch.abs(out)

    def __repr__(self):
        class_name = self.__class__.__name__

        out = f"{class_name}(order1={self.order1.item()}, order2={self.order2.item()})"
        return out


class FrFTPool(nn.Module):
    def __init__(self, order: float = 1) -> None:
        super(FrFTPool, self).__init__()

        self.order1 = nn.Parameter(torch.tensor(order, dtype=torch.float32))
        self.order2 = nn.Parameter(torch.tensor(order, dtype=torch.float32))

    def forward(self, x):
        #
        out = frft(x, self.order1, dim=-1)
        out = frft(out, self.order2, dim=-2)

        *_, H, W = out.size()

        st_H = H // 4
        end_H = H - st_H

        st_W = W // 4
        end_W = W - st_W

        out = out[..., st_H:end_H, st_W:end_W]

        #
        out = frft(out, -self.order1, dim=-1)
        out = frft(out, -self.order2, dim=-2)

        return torch.abs(out)

    def __repr__(self):
        class_name = self.__class__.__name__

        out = f"{class_name}(order1={self.order1.item()}, order2={self.order2.item()})"
        return out


def main():
    pass


if __name__ == "__main__":
    main()

# %%
