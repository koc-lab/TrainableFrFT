'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch_frft.dfrft_module import dfrft
from torch_frft.frft_module import frft


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
       
      



        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2((self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, model_name, num_classes=10,order: float = 1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.model_name=model_name

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.order1 = nn.Parameter(torch.tensor(order, dtype=torch.float32))
        self.order2 = nn.Parameter(torch.tensor(order, dtype=torch.float32))
      
       
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            
            if "DFRFT" in self.model_name:
                        out = dfrft(out, self.order1, dim=-1)
                        out = dfrft(out, self.order2, dim=-2)
                        out=torch.abs(out)

            elif "FRFT" in self.model_name:
                        out = frft(out, self.order1, dim=-1)
                        out = frft(out, self.order2, dim=-2)
                        out=torch.abs(out)  
            elif "FFT" in self.model_name:
                        out= torch.abs(torch.fft.fft2(out,norm="ortho"))

            out = self.layer3(out)
            out = self.layer4(out)  
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

            return out
   
    def __repr__(self):
        class_name = self.__class__.__name__

        out = f"{class_name}(order1={self.order1.item()}, order2={self.order2.item()})"
        return out


    # TODO: add a method to get the fractional orders of the model
    def get_frac_orders(self):
        d = {}
      
        if "DFRFT" in self.model_name or "FRFT" in self.model_name:
                order_1, order_2 = self.order1.item(), self.order2.item()   
                d["frac"] = (order_1, order_2) 
                # TODO: We can add module type operation to put values in desired ranges

  
        return d


def resnet_models(model_name,n_class):

    if "ResNet18" in model_name:
        return ResNet(BasicBlock, [2, 2, 2, 2],model_name=model_name,num_classes=n_class)
    elif "ResNet34" in model_name:
        return ResNet(BasicBlock, [3, 4, 6, 3],model_name=model_name,num_classes=n_class)
    elif "ResNet50" in model_name:
        return ResNet(Bottleneck, [3, 4, 6, 3],model_name=model_name,num_classes=n_class)


'''
def ResNet101(num_class,domain):
    return ResNet(Bottleneck, [3, 4, 23, 3],num_class,domain)


def ResNet152(num_class):
    return ResNet(Bottleneck, [3, 8, 36, 3],num_class)
'''


if __name__=="__main__":
     model=  resnet_models(model_name="ResNet50",n_class=10)
     data= torch.rand(50,3,32,32)
     target= model(data)
     print(target.shape)