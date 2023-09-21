import torch
import torch.nn as nn
from torchvision import models
import os
import sys
import numpy as np
from torch.nn import init


from models.custom_frft_layers import FrFTPool, DFrFTPool, FFTPool

class ResNet(nn.Module):
    
    def __init__(self,model_name, pre_trained=True, n_class=200, model_choice=50):
        super(ResNet, self).__init__()
        expansion=4
        if model_choice ==18 or model_choice==34:
            expansion=1
        self.n_class = n_class
        self.model_name=model_name
       
        self.base_model = self._model_choice(pre_trained, model_choice)
       
        if "DFRFT" in self.model_name:
            self.base_model.avgpool = DFrFTPool()
            self.base_model.fc = nn.Linear(512*expansion*4*4, n_class)
        elif "FRFT" in self.model_name:
            self.base_model.avgpool = FrFTPool()
            self.base_model.fc = nn.Linear(512*expansion*4*4, n_class)
        elif "FFT" in self.model_name:
            self.base_model.avgpool = FFTPool()
            self.base_model.fc = nn.Linear(512*expansion*4*4, n_class)
        else: 
             self.base_model.avgpool=nn.AdaptiveAvgPool2d((1,1))
             self.base_model.fc = nn.Linear(512*expansion, n_class)
                
         #this part is crucial!!!!!!!!!!!!
        self.base_model.fc.apply(weight_init_kaiming)
       
       
        self.features1 = torch.nn.Sequential(
                self.base_model.conv1,
                self.base_model.bn1,
                self.base_model.relu,
                self.base_model.maxpool,
                self.base_model.layer1,
                self.base_model.layer2, 
                self.base_model.layer3, 
                self.base_model.layer4)
    


    def forward(self, x):
        
        N=x.size(0)
        x=self.features1(x)
        x=self.base_model.avgpool(x)
        x=x.contiguous().view(x.size(0), -1)
        x=self.base_model.fc(x)
        assert x.size() == (N, self.n_class)
        return x

    def _model_choice(self, pre_trained, model_choice):
        
        if model_choice == 50:
            return models.resnet50(pretrained=pre_trained)
        elif model_choice == 101:
            return models.resnet101(pretrained=pre_trained)
        elif model_choice == 152:
            return models.resnet152(pretrained=pre_trained)
        elif model_choice==18:
            return models.resnet18(pretrained=pre_trained)
        elif model_choice==34:
            return models.resnet34(pretrained=pre_trained)
   
    def get_frac_orders(self):
        d = {}
      
        if "DFRFT" in self.model_name or "FRFT" in self.model_name:
                order_1, order_2 = self.base_model.avgpool.order1.item(), self.base_model.avgpool.order2.item()   
                d["frac"] = (order_1, order_2) 
                # TODO: We can add module type operation to put values in desired ranges

  
        return d


def resnet_models(model_name,n_class):
        
        if  "ResNet18" in model_name:
               return ResNet(model_choice=18,model_name=model_name,n_class=n_class)
        elif "ResNet34" in model_name:  
               return ResNet(model_choice=34,model_name=model_name,n_class=n_class)
        elif "ResNet50" in model_name:
              return ResNet(model_choice=50,model_name=model_name,n_class=n_class)
        elif "ResNet101" in model_name:
              return ResNet(model_choice=101,model_name=model_name,n_class=n_class)
        elif "ResNet152" in model_name:
              return ResNet(model_choice=152,model_name=model_name,n_class=n_class)
                  

def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
        #init.constant_(m.bias.data, 0.0)



if __name__=="__main__":
     model=resnet_models("ResNet50_FRFT",200)
     print(model)