import time

from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models_x import *
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
