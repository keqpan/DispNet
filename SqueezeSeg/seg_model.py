import numpy as np
import torch
from os.path import join
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#Convolution + BatchNorm  ReLU
class myconv(nn.Module):
    def __init__(self, n_in, n_out,n_ker,n_str=1,pad=0,batch=True,relu=True):
        super(myconv, self).__init__()
        self.bbatch = batch;
        self.brelu = relu;
        self.conv = nn.Conv2d(n_in, n_out, n_ker,stride=n_str,padding=pad);
        self.batch = nn.BatchNorm2d(n_out);
        self.relu = nn.ReLU(inplace=True);

    def forward(self, x):
        x = self.conv(x);
        if self.bbatch == True:
            x = self.batch(x);
        if self.brelu == True:
            x = self.relu(x);
        return x

#Deonvolution + BatchNorm  ReLU
class mydeconv(nn.Module):
    def __init__(self, n_in, n_out,n_ker,n_str=1,pad=1,batch=True,relu=True,opad=0):
        super(mydeconv, self).__init__()
        self.bbatch = batch;
        self.brelu = relu;
        self.deconv = nn.ConvTranspose2d(in_channels=n_in, out_channels=n_out, kernel_size=n_ker,stride=n_str,padding=pad,output_padding=opad);
        self.batch = nn.BatchNorm2d(n_out);
        self.relu = nn.ReLU(inplace=True);

    def forward(self, x):
        x = self.deconv(x);
        if self.bbatch == True:
            x = self.batch(x);
        if self.brelu == True:
            x = self.relu(x);
        return x

#Fire module
class myfire(nn.Module):
    def __init__(self,n_in,s11,e11,e33):
        super(myfire, self).__init__()
        self.convinit = myconv(n_in, s11, 1);
        self.conv1 = myconv(s11, e33, 3, pad=1);
        self.conv2 = myconv(s11, e11, 1);
        
    def forward(self, x):
        x = self.convinit(x);
        x1 = self.conv1(x);
        x2 = self.conv2(x);
        return torch.cat([x1, x2],dim=1);

#Fire upscaling module
class myfireup(nn.Module):
    def __init__(self,n_in,s11,e11,e33,scale=(1,2),outpad=0):
        super(myfireup, self).__init__()
        self.convinit = myconv(n_in, s11, 1);
        self.decup = mydeconv(s11,s11,3,n_str=scale,opad=outpad);
        self.conv1 = myconv(s11, e33, 3, pad=1);
        self.conv2 = myconv(s11, e11, 1);
        
    def forward(self, x):
        x = self.convinit(x);
        x = self.decup(x);
        x1 = self.conv1(x);
        x2 = self.conv2(x);
        return torch.cat([x1, x2],dim=1);

#Network itself
class SqueezeSeg(nn.Module):
    def __init__(self):
        super(SqueezeSeg, self).__init__()
        self.conv1a = myconv(6,64,3,2,1);
        self.conv1b = myconv(6,64,1);
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2),padding=(1,1));
        self.fire2 = myfire(64,16,64,64);
        self.fire3 = myfire(128,16,64,64);
        self.pool3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2),padding=(1,1));
        self.fire4 = myfire(128,32,128,128);
        self.fire5 = myfire(256,32,128,128);
        self.pool5 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2),padding=(1,1));
        self.fire6 = myfire(256,48,192,192);
        self.fire7 = myfire(384,48,192,192);
        self.fire8 = myfire(384,64,256,256);
        self.fire9 = myfire(512,64,256,256);
        self.fire10dec = myfireup(512,64,128,128,scale=(2,2),outpad=(1,1));
        self.fire11dec = myfireup(256,32,64,64,scale=(2,2),outpad=(0,1));
        self.fire12dec = myfireup(128,16,32,32,scale=(2,2),outpad=(1,1));
        self.fire13dec = myfireup(64,16,32,32,scale=(2,2),outpad=(1,1));
        self.conv14 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3,stride=1,padding=1);
        self.relu = nn.LeakyReLU(inplace=True);

    def forward(self, x):
        x1 = self.conv1b(x);
        #return x1; #64x540x960
        x2 = self.conv1a(x);
        #return x2; #64x270x480
        x = self.pool1(x2);
        #return x;  #64x135x240
        x3 = self.fire2(x);
        #return x3; #128x135x240
        x = self.fire3(x3);
        #return x;  #128x135x240
        x = self.pool3(x);
        #return x;  #128x68x120
        x4 = self.fire4(x);
        #return x4; #256x68x120
        x = self.fire5(x4);
        #return x;  #256x68x120
        x = self.pool5(x);
        #return x;  #256x34x60
        x = self.fire6(x);
        #return x;  #384x34x60
        x = self.fire7(x);
        #return x;  #384x34x60
        x = self.fire8(x);
        #return x;  #512x34x60
        x = self.fire9(x);
        #return x;  #512x34x60
        x = self.fire10dec(x);
        #return x;  #256x68x120
        x = x + x4;
        #return x;  #256x68x120
        x = self.fire11dec(x);
        #return x;  #128x135x240
        x = x + x3;
        #return x;  #128x135x240
        x = self.fire12dec(x);
        #return x;  #64x270x480
        x = x + x2;
        #return x;  #64x270x480
        x = self.fire13dec(x);
        #return x;  #64x540x960
        x = x + x1;
        #return x;  #64x540x960
        x = self.conv14(x);
        #return x;  #4x540x960
        x = self.relu(x);
        return x;