import torch
import torch.nn as nn
from torchvision import models
import os
import sys
import numpy as np
from torch.nn import init


from models.custom_frft_layers import FrFTPool, DFrFTPool, FFTPool



class VGG16(nn.Module):

    def __init__(self,classes,model_name):
        
        super(VGG16, self).__init__()
        self.base_model=models.vgg16(pretrained=True)
        self.classes=classes
        self.model_name=model_name
       
        flat_dim=3*3
        if "DFRFT" in self.model_name:
            self.base_model.avgpool = DFrFTPool()
            
        
        elif "FRFT" in self.model_name:
            self.base_model.avgpool = FrFTPool()

        elif "FFT" in self.model_name:
            self.base_model.avgpool = FFTPool()
        
        else:
            self.base_model.avgpool= nn.AdaptiveAvgPool2d((1,1))
            flat_dim=1*1

        
        
        self.base_model.classifier[0]=nn.Linear(in_features=512*flat_dim, out_features=4096, bias=True)
        self.base_model.classifier[6]= nn.Linear(in_features=4096, out_features=self.classes, bias=True) #4096 here is from original pretrained model with imagenet
        self.base_model.classifier[6].apply(weight_init_kaiming)
        self.base_model.classifier[0].apply(weight_init_kaiming)
        
        

    def forward(self,x):
          return self.base_model(x)
    
class VGG13(nn.Module):

    def __init__(self,classes,model_name):
        
        super(VGG13, self).__init__()
        self.base_model=models.vgg13(pretrained=True)
        self.classes=classes
        self.model_name=model_name
       
        flat_dim=3*3
        if "DFRFT" in self.model_name:
            self.base_model.avgpool = DFrFTPool()
            
        
        elif "FRFT" in self.model_name:
            self.base_model.avgpool = FrFTPool()

        elif "FFT" in self.model_name:
            self.base_model.avgpool = FFTPool()
        
        else:
            self.base_model.avgpool= nn.AdaptiveAvgPool2d((1,1))
            flat_dim=1*1

        
        
        self.base_model.classifier[0]=nn.Linear(in_features=512*flat_dim, out_features=4096, bias=True)
        self.base_model.classifier[6]= nn.Linear(in_features=4096, out_features=self.classes, bias=True) #4096 here is from original pretrained model with imagenet
        self.base_model.classifier[6].apply(weight_init_kaiming)
        self.base_model.classifier[0].apply(weight_init_kaiming)
        
        

    def forward(self,x):
          return self.base_model(x)
    

class VGG11(nn.Module):

    def __init__(self,classes,model_name):
        
        super(VGG11, self).__init__()
        self.base_model=models.vgg11(pretrained=True)
        self.classes=classes
        self.model_name=model_name
       
        flat_dim=3*3
        if "DFRFT" in self.model_name:
            self.base_model.avgpool = DFrFTPool()
            
        
        elif "FRFT" in self.model_name:
            self.base_model.avgpool = FrFTPool()

        elif "FFT" in self.model_name:
            self.base_model.avgpool = FFTPool()
        
        else:
            self.base_model.avgpool= nn.AdaptiveAvgPool2d((1,1))
            flat_dim=1*1

        
        
        self.base_model.classifier[0]=nn.Linear(in_features=512*flat_dim, out_features=4096, bias=True)
        self.base_model.classifier[6]= nn.Linear(in_features=4096, out_features=self.classes, bias=True) #4096 here is from original pretrained model with imagenet
        self.base_model.classifier[6].apply(weight_init_kaiming)
        self.base_model.classifier[0].apply(weight_init_kaiming)
        
        

    def forward(self,x):
          return self.base_model(x)
    




        

def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
      


#check from which maxpooling layers you want
if __name__== "__main__":

        # Step 1: Load pretrained VGG-16 model
    #vgg16 = models.vgg16(pretrained=True)
    vgg16= VGG11(classes=200,model_name="VGG13_DFRFT")

        # Step 2: Print the model to inspect its architectur
    print(vgg16)

