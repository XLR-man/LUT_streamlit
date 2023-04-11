import argparse
import time

from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models_x import *
from datasets import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=210, help="epoch to load the saved checkpoint")
parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--model_dir", type=str, default="LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10", help="directory of saved models")
opt = parser.parse_args()
opt.model_dir = opt.model_dir + '_' + opt.input_color_space

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
LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.model_dir, opt.epoch))
LUT0.load_state_dict(LUTs["0"])
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])

LUT0.eval()
LUT1.eval()
LUT2.eval()
classifier.load_state_dict(torch.load("saved_models/%s/classifier_%d.pth" % (opt.model_dir, opt.epoch)))
classifier.eval()

def generator(img):

    pred = classifier(img).squeeze()
    print("weight:",pred)
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT #+ pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    combine_A = img.new(img.size())
    _, combine_A = trilinear_(LUT,img)

    return combine_A

def runforstreamlit(image):

    out_dir = "test_images/%s_%d" % (opt.model_dir, opt.epoch)
    os.makedirs(out_dir, exist_ok=True)

    # Load the image
    img = TF.to_tensor(image)
    img = img.unsqueeze(0)
    real_A = Variable(img.type(Tensor))
    fake_B = generator(real_A)
    save_image(fake_B, os.path.join(out_dir,"1.png"), nrow=1, normalize=False)
    result = Image.open(os.path.join(out_dir,"1.png"))
    return result
