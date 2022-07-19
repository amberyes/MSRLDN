from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable


class SelectiveResidualBlock(nn.Module):
    def __init__(self,input_channel):
        super(SelectiveResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1)
        self.nl1=nn.BatchNorm2d(input_channel)
        self.conv2 = nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1)
        self.nl2=nn.BatchNorm2d(input_channel)

        self.ac=nn.relu(inplace=True)

        self.a=nn.Parameter(data=torch.ones(1))
        self.b=nn.Parameter(data=torch.ones(1))

    def forward(self,x):
        r=self.ac(self.nl1(self.conv1(x)))
        r=self.nl2(self.conv2(r))
        out=self.ac(r.mul(self.b)+x.mul(self.a))
        return out

class EncodeBlock(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(EncodeBlock,self).__init__()
        self.srb1 = SelectiveResidualBlock(input_channel)
        self.srb2 = SelectiveResidualBlock(input_channel)
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1)
        
        self.ac=nn.relu(inplace=True)

    def forward(self,x):
        out=self.ac(self.conv(self.srb2(self.srb1(x))))
        return out

class DecodeBlock(nn.Module):
    def __init__(self,input_channel,output_channel ):
        super(DecodeBlock,self).__init__()
        self.up=nn.PixelShuffle(2)
        self.srb1 = SelectiveResidualBlock(output_channel)
        self.srb2 = SelectiveResidualBlock(output_channel)
    

    def forward(self,x,skip):
        out=self.up(x)
        out=torch.cat((out,skip),dim=1)
        out=self.srb2(self.srb1(self.up(out)))

        return out

class CovBlock(nn.Module):
    def __init__(self,input_channel,output_channel, stride=1):
        super(CovBlock,self).__init__()
        self.cov=nn.n.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1)
        self.nl=nn.BatchNorm2d(output_channel)
    def forward(self,x):
        
        out=self.bn(self.cov(x))

        return out




class J_net(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, activation=nn.ReLU(inplace=True)):
        super(J_net, self).__init__()

        self.cov10=CovBlock(in_channels, 32)
        self.ac10=activation
        self.cov11=CovBlock(32, 32)

        self.cov20=CovBlock(in_channels, 64)
        self.ac20=activation
        self.cov21=CovBlock(64, 32)

        self.cov30=CovBlock(in_channels, 128)
        self.ac30=activation
        self.cov31=CovBlock(128, 128)

        self.cov40=CovBlock(in_channels, 256)
        self.ac40=activation
        self.cov41=CovBlock(256, 256)

        # Encoder
        # x_org
        self.enc11=EncodeBlock(32,64)
        self.enc12=EncodeBlock(64,128)
        self.enc13=EncodeBlock(128,256)

        # x/2
        self.enc22=EncodeBlock(64,128)
        self.enc23=EncodeBlock(128,256)

        # x/4
        self.enc33=EncodeBlock(128,256)

        # bottle
        self.cov34=nn.n.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.ac34=activation
        self.rsb34=SelectiveResidualBlock(512)

        # Decoder
        self.dec44=DecodeBlock(512,256)
        self.dec34=DecodeBlock(256,128)
        self.dec24=DecodeBlock(128,64)

        self.cov15=CovBlock(64, 32)
        self.ac15=activation
        self.cov16=CovBlock(32, 3)

        self.cov25=CovBlock(128, 64)
        self.ac25=activation
        self.cov26=CovBlock(64, 32)

        self.cov35=CovBlock(256, 128)
        self.ac35=activation
        self.cov36=CovBlock(128, 3)

        self.cov45=CovBlock(512, 256)
        self.ac45=activation
        self.cov46=CovBlock(256, 3)

        self.ac5=activation
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x_org=x
        x_2=F.interpolate(x,0.5)
        x_4=F.interpolate(x,0.25)
        x_8=F.interpolate(x,0.125)

        x10=self.cov11(self.ac10(self.cov10(x_org)))
        x11=self.enc11(x10)
        x12=self.enc12(x11)
        x13=self.enc13(x12)

        x20=self.cov21(self.ac20(self.cov20(x_2)))
        x21=self.enc22(x20)
        x22=self.enc23(x21)

        x30=self.cov31(self.ac30(self.cov30(x_4)))
        x31=self.enc33(x30)

        x40=self.cov41(self.ac40(self.cov40(x_8)))

        x41=torch.cat((x13,x22,x31,x40),dim=1)
        x42=self.rsb34(self.ac34(self.cov34(x41)))
        out3=self.ac5(self.cov46(self.ac45(self.cov45(x42))))

        x2_o=self.dec44(x42,(x30+x21+x12))
        out2=self.ac5(self.cov36(self.ac35(self.cov35(x2_o))))

        x1_o=self.dec34(x2_o,(x20+x11))
        out1=self.ac5(self.cov26(self.ac25(self.cov25(x1_o))))

        x0_o=self.dec34(x1_o,x10)
        out0=self.ac5(self.cov16(self.ac15(self.cov15(x1_o))))
        
        return out0,out1,out2,out3


        





        
        
        

        
