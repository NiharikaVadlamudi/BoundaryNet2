import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
import torchvision.ops
from torch import nn



# CUDA Device Settings 
# For MCNN Training , always assign 1 GPU only - Preferably GPU 0 
# Dataloader can be sent to GPU 1.0 

if torch.cuda.is_available():
    print('------------------- Device Information -------------------------------')
    print('Allocated GPUs : {} , CPUs : {}'.format(torch.cuda.device_count(),os.cpu_count()))
    print('__CUDNN VERSION  : {} '.format(torch.backends.cudnn.version()))
    print('__Number CUDA Devices  : {}'.format(torch.cuda.device_count()))
    if(torch.cuda.device_count()<1):
        print('Only {} GPUs  allocated , minimum required ! '.format(torch.cuda.device_count()))
    else: 
        # Encoder will have only device 1 .
        device0 = torch.device("cuda:0")
        # device0=torch.device("cuda:1")
else:
    print(" NO GPU ALLOCATED , Exiting . ")
    sys.exit()

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv1(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv1,self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class Model(nn.Module):
    def __init__(self, input_height, input_width, input_channels):
        super(Model, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels


        # ----------- RESNET initialization  -----------
        self.layer_d0 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, stride=2, bias=True),
                                      nn.ReLU()).to(device0)

        self.layer_d10 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1, stride=1, bias=True),
                                      nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1, stride=1, bias=True)).to(device0)
        self.layer_d10s = nn.Conv2d(8, 16, 1, stride=1).to(device0)
        self.relu_d10 = nn.ReLU().to(device0)


        # ResNet 32 
        self.layer_d12 = nn.Sequential(nn.Conv2d(16, 32,3, padding=1, stride=1, bias=True),
                                      nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1, stride=1, bias=True)).to(device0)
        self.layer_d12s = nn.Conv2d(16, 32, 1, stride=1).to(device0)
        self.relu_d12 = nn.ReLU().to(device0)


        # Resnet-64 
        self.layer_d14 = nn.Sequential(nn.Conv2d(32, 64, 3,padding=1, stride=1, bias=True),
                                      nn.ReLU(), nn.Conv2d(64, 64, 3,padding=1, stride=1, bias=True)).to(device0)
        self.layer_d14s = nn.Conv2d(32, 64, 1, stride=1).to(device0)
        self.relu_d14 = nn.ReLU().to(device0)


        # Resnet - 128 
        self.layer_d15 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=1, bias=True),
                                      nn.ReLU(), nn.Conv2d(128, 128, 3,padding=1,stride=1, bias=True)).to(device0)
        self.layer_d15s = nn.Conv2d(64, 128, 1, stride=1).to(device0)
        self.relu_d15 = nn.ReLU().to(device0)


        self.adapt_avg = nn.AdaptiveMaxPool2d((2,2)).to(device0)


        # ----------- SAG initialization -----------

        self.Up5 = up_conv1(ch_in=128,ch_out=64).to(device0)
        self.Att5 = Attention_block(F_g=64,F_l=64,F_int=32).to(device0)
        self.Up_conv5 = conv_block(ch_in=128, ch_out=64).to(device0)

        self.Up4 = up_conv1(ch_in=64,ch_out=32).to(device0)
        self.Att4 = Attention_block(F_g=32,F_l=32,F_int=16).to(device0)
        self.Up_conv4 = conv_block(ch_in=64, ch_out=32).to(device0)
        
        self.Up3 = up_conv1(ch_in=32,ch_out=16).to(device0)
        self.Att3 = Attention_block(F_g=16,F_l=16,F_int=8).to(device0)
        self.Up_conv3 = conv_block(ch_in=32, ch_out=16).to(device0)

        self.Up2 = up_conv1(ch_in=16,ch_out=8).to(device0)
        self.Att2 = Attention_block(F_g=8,F_l=8,F_int=4).to(device0)
        self.Up_conv2 = conv_block(ch_in=16, ch_out=8).to(device0)

        # ----------- Mask Decoder -----------
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),nn.ReLU()).to(device0)
        self.mask_classifier = nn.Sequential(nn.Conv2d(8, 1, 1, padding=0, bias=True)).to(device0)


        # # ----------- Linear layers - region classifier -----------
        self.fc1 = nn.Sequential(nn.Linear(512, 128, bias=True), nn.ReLU()).to(device0)
        self.fc2 = nn.Sequential(nn.Linear(128, 64, bias=True),nn.ReLU()).to(device0)
        self.fc3 = nn.Sequential(nn.Linear(64, 9, bias=True)).to(device0)


    def forward(self, input_img):
        
        # resnet backbone

        output1 = self.layer_d0(input_img.to(device0))

        output_d10 = self.layer_d10(output1.to(device0))
        output_d10s = self.layer_d10s(output1.to(device0))
        output = output_d10 + output_d10s
        output2 = self.relu_d10(output.to(device0))

        output_d12 = self.layer_d12(output2.to(device0))
        output_d12s = self.layer_d12s(output2.to(device0))
        output = output_d12 + output_d12s
        output3 = self.relu_d12(output.to(device0))

        output_d14 = self.layer_d14(output3.to(device0))
        output_d14s = self.layer_d14s(output3.to(device0))
        output = output_d14 + output_d14s
        output4 = self.relu_d14(output.to(device0))


        output_d15 = self.layer_d15(output4.to(device0))
        output_d15s = self.layer_d15s(output4.to(device0))
        output = output_d15 + output_d15s
        output5 = self.relu_d15(output.to(device0))

        df48 = output5

        ## SAG Blocks

        d5 = self.Up5(output5.to(device0))
        output4 = self.Att5(g=d5.to(device0),x=output4.to(device0))
        d5 = torch.cat((output4.to(device0),d5.to(device0)),dim=1)        
        d5 = self.Up_conv5(d5.to(device0))
        
        d4 = self.Up4(d5.to(device0))
        output3 = self.Att4(g=d4.to(device0),x=output3.to(device0))
        d4 = torch.cat((output3.to(device0),d4.to(device0)),dim=1)
        d4 = self.Up_conv4(d4.to(device0))

        d3 = self.Up3(d4.to(device0))
        output2 = self.Att3(g=d3.to(device0),x=output2.to(device0))
        d3 = torch.cat((output2.to(device0),d3.to(device0)),dim=1)
        d3 = self.Up_conv3(d3.to(device0))

        d2 = self.Up2(d3.to(device0))
        output1 = self.Att2(g=d2.to(device0),x=output1.to(device0))
        d2 = torch.cat((output1.to(device0),d2.to(device0)),dim=1)
        d2 = self.Up_conv2(d2.to(device0))

        df4 = torch.cat((d2, d3, d4, d5), dim = 1)

        #Decoder
        deconv_1 = self.deconv1(d2.to(device0))

        # Mask Logits
        mask_logits = self.mask_classifier(deconv_1.to(device0))

        # class branch
        class_logits = self.adapt_avg(df48.to(device0))
        class_logits = class_logits.reshape(class_logits.shape[0], -1).to(device0)
        class_prob = self.fc1(class_logits.to(device0))
        class_prob = self.fc2(class_prob.to(device0))
        class_prob = self.fc3(class_prob.to(device0))
        
    
        return df4.to(device0) , mask_logits.to(device0) , class_prob.to(device0)

