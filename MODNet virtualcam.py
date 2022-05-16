import cv2
import torch
from torch import nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pyvirtualcam
from pyvirtualcam import PixelFormat
import torchvision

vgg = torchvision.models.vgg16(True)
vgg.cuda()
for param in vgg.parameters():
    param.requires_grad = False
class Sp(nn.Module):
  def __init__(self,vgg16,M=4):
    super(Sp,self).__init__()
    self.vgg = vgg16

    self.conv1 = nn.Conv2d(3,32,3,padding= 'same')
    self.act1 = nn.Sigmoid()
    self.max1 = nn.MaxPool2d(4)

    self.conv2 = nn.Conv2d(32,32,3,padding= 'same')
    self.act2 = nn.Sigmoid()
    self.max2 = nn.MaxPool2d(2)

    self.upsample1 = nn.Upsample(scale_factor = 2) #16x16
    self.convtrans1 = nn.ConvTranspose2d(512,128,(3,3),padding = 1)
    self.sig1 = nn.Sigmoid()
    self.upsample2 = nn.Upsample(scale_factor = 2) #32x32
    self.convtrans2 = nn.ConvTranspose2d(160,1,(3,3),padding = 1)
    self.sig2 = nn.Sigmoid()
    #self.thresh = nn.Threshold(0.5,0)

  def forward(self,X):
    #pretrained vgg for low level features
    V = self.vgg(X)
    
    #high level features
    X = self.conv1(X)
    X = self.act1(X)
    X = self.max1(X)
    X = self.conv2(X)
    X = self.act2(X)
    I = self.max2(X)


    
    X = self.upsample1(V)
    X = self.convtrans1(X)
    X = self.sig1(X)
    X = self.upsample2(X)

    #high + low level features
    X = torch.cat([X,I],1)
    X = self.convtrans2(X)
    X = self.sig2(X)



    return X

class Dp(nn.Module):
  def __init__(self):
    super(Dp,self).__init__()
    self.conv1 = nn.Conv2d(3,64,(3,3),padding = 'same')
    self.max1 = nn.MaxPool2d(2) #128x128
    self.act1 = nn.Sigmoid()


    self.upsample_S = nn.Upsample(scale_factor = 2) 

    self.convtrans1 = nn.ConvTranspose2d(65,64,(3,3),padding = 1)
    self.act2 = nn.Sigmoid()


    self.upsample1 = nn.Upsample(scale_factor = 2) #256x256
    self.convtrans2 = nn.ConvTranspose2d(64,8,(3,3),padding = 1)
    self.act3 = nn.Sigmoid()


    self.upsample_Dp = nn.Upsample(scale_factor = 2) #512x512
    self.convtrans3 = nn.ConvTranspose2d(11,1,(3,3),padding=1)
    self.sig1 = nn.Sigmoid()
    #self.thresh = nn.Threshold(0.5,0)


  def forward(self,X,S):
    skip_conn = X

    X = self.conv1(X)
    X = self.act1(X)
    X = self.max1(X)

    S = self.upsample_S(S)
    X = torch.cat([X,S],1)

    X = self.convtrans1(X)
    X = self.act2(X)
    X = self.upsample1(X)
    X = self.convtrans2(X)
    X = self.act3(X)

    D_p = torch.cat([X,skip_conn],1)
    D_p = self.upsample_Dp(D_p)
    D_p = self.convtrans3(D_p)
    D_p = self.sig1(D_p)
    #D_p = self.thresh(D_p)

    self.D_p = D_p.clone()


    return X
  
  def getDp(self):
    return self.D_p

class Fp(nn.Module):
  def __init__(self):
    super(Fp,self).__init__()

    self.upsample1 = nn.Upsample(scale_factor = 2)
    self.convtrans1 = nn.ConvTranspose2d(1,16,(3,3),padding = 1)
    self.act1 = nn.Sigmoid()


    self.upsample2 = nn.Upsample(scale_factor = 2)
    self.convtrans2 = nn.ConvTranspose2d(24,16,(3,3),padding = 1)
    self.act2 = nn.Sigmoid()


    self.upsample3 = nn.Upsample(scale_factor = 2)
    self.convtrans3 = nn.ConvTranspose2d(16,1,(3,3),padding = 1)
    self.sig1 = nn.Sigmoid()
    #self.thresh = nn.Threshold(0.5,0)


  def forward(self,S,D):
    S = self.upsample1(S)
    S = self.convtrans1(S)
    S = self.act1(S)

    S = self.upsample2(S)
    S = torch.cat([S,D],1)
    S = self.convtrans2(S)
    S = self.act2(S)

    S = self.upsample3(S)
    S = self.convtrans3(S)
    S = self.sig1(S)
    #S = self.thresh(S)
    return S
class MODNet(nn.Module):
  def __init__(self,vgg16):
    super(MODNet,self).__init__()
    self.S = Sp(vgg16)
    self.D = Dp()
    self.F = Fp()
    self.max1 = nn.MaxPool2d(2)
    self.upsample1 = nn.Upsample(scale_factor = 2)

  def forward(self,X):
    I = X
    X = self.S(X)

    I = self.max1(I)

    D = self.D(I,X)
    F = self.F(X,D)
    details = self.D.getDp()
    return [F,X,details]
def pred(img):
  img = cv2.resize(img,(256,256))
  img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  img = np.array(img)/255
  img1 = img
  
  img = torch.tensor(img,dtype = torch.float).permute(2,0,1).float().cuda()
  out = model(img.unsqueeze(0))
  return out[0].cpu().detach().squeeze(2).numpy()[0][0],img1

cp = cv2.VideoCapture(0)
cam2 = pyvirtualcam.Camera(256, 256, 5, fmt=PixelFormat.RGB)
model = torch.load("6-F")
model.cuda()
cp.set(cv2.CAP_PROP_FPS,20)

while True:
  i,img = cp.read()
  out = pred(img)
  out[0][out[0]<1] = 0
  out[0][out[0]>=1] = 1
  outm = np.repeat(out[0][:,:,np.newaxis],3,2)

  cam2.send(np.array(out[1]*outm*255,np.uint8))
  cam2.sleep_until_next_frame()
cp.release()