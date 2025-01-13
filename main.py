import torch 
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet # model itself
from realesrgan import RealESRGANer # framework that uses it

model_path = 'RealESRGAN_x4plus.pth' # path to the model

state_dict = torch.load(model_path, map_location=torch.device('cpu'))['params_ema'] # load the model    

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4) # create an instance of the model