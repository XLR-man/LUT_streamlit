import streamlit as st
import subprocess
import sys
import torch

# import image_adaptive_lut_evaluation
import time

from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F

# from models_x import *
st.set_page_config(layout="wide", page_title="Low Light Image Enhancement")
st.write("## Enhance your low light dsaimageh121")

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np
import math
import trilinear

def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#        Discriminator
##############################

def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(nn.BatchNorm2d(out_filters))

    return layers

class Classifier(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            #*discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim-1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)
        
        assert 1 == trilinear.forward(lut, 
                                      x, 
                                      output,
                                      dim, 
                                      shift, 
                                      binsize, 
                                      W, 
                                      H, 
                                      batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        
        ctx.save_for_backward(*variables)
        
        return lut, output
    
    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
            
        assert 1 == trilinear.backward(x, 
                                       x_grad, 
                                       lut_grad,
                                       dim, 
                                       shift, 
                                       binsize, 
                                       W, 
                                       H, 
                                       batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)

st.set_page_config(layout="wide", page_title="Low Light Image Enhancement")
st.write("## Enhance your low light ds211")
from datasets import *


epoch = 210
dataset_name = "fiveK"
input_color_space ="sRGB"
model_dir = "LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10"

model_dir = model_dir + '_' + input_color_space

# use gpu when detect cuda
cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

criterion_pixelwise = torch.nn.MSELoss()
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()

classifier = Classifier()
trilinear_ = TrilinearInterpolation() 

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()

# Load pretrained models
LUTs = torch.load("https://github.com/XLR-man/LUT_streamlit/tree/master/saved_models/%s/LUTs_%d.pth" % (model_dir, epoch))
LUT0.load_state_dict(LUTs["0"])
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])

LUT0.eval()
LUT1.eval()
LUT2.eval()
classifier.load_state_dict(torch.load("https://github.com/XLR-man/LUT_streamlit/tree/master/saved_models/%s/classifier_%d.pth" % (model_dir, epoch)))
classifier.eval()

def generator(img):

    pred = classifier(img).squeeze()
    print("weight:",pred)
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT

    combine_A = img.new(img.size())
    _, combine_A = trilinear_(LUT,img)

    return combine_A

def runforstreamlit(image):

    out_dir = "https://github.com/XLR-man/LUT_streamlit/tree/master/test_images/%s_%d" % (model_dir, epoch)
    os.makedirs(out_dir, exist_ok=True)

    # Load the image
    img = TF.to_tensor(image)
    img = img.unsqueeze(0)
    real_A = Variable(img.type(Tensor))
    fake_B = generator(real_A)
    save_image(fake_B, os.path.join(out_dir,"1.png"), nrow=1, normalize=False)
    result = Image.open(os.path.join(out_dir,"1.png"))
    return result

# import evaluation_LOL
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="Low Light Image Enhancement")

st.write("## Enhance your low light image")
st.write(
    ":dog: Try uploading an image to watch the enhanced image. Full quality images can be downloaded from the sidebar.:grin:"
)
st.sidebar.write("## Upload and download :gear:")


# # Download the fixed image
# def convert_image(img):
#     buf = BytesIO()
#     img.save(buf, format="PNG")
#     byte_im = buf.getvalue()
#     return byte_im


# def fix_image(upload):
#     image = Image.open(upload)
#     col1.write("Original Image :camera:")
#     col1.image(image)

#     fixed = image_adaptive_lut_evaluation.runforstreamlit(image)
#     col2.write("Ehanced Image :wrench:")
#     col2.image(fixed)
#     st.sidebar.markdown("\n")
#     st.sidebar.download_button("Download ehanced image", convert_image(fixed), "ehanced.png", "image/png")


# col1, col2 = st.columns(2)
# my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if my_upload is not None:
#     fix_image(upload=my_upload)
# else:
#     fix_image("https://github.com/XLR-man/LUT_streamlit/tree/master/demo_images/sRGB/a1629.jpg")
