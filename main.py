import torch 
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet # model itself
from realesrgan import RealESRGANer # framework that uses it

model_path = 'RealESRGAN_x4plus.pth' # path to the model

state_dict = torch.load(model_path, map_location=torch.device('cpu'))['params_ema'] # load the model    

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4) # create an instance of the model
model.load_state_dict(state_dict, strict=True) # load the state dict into the model


upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    pre_pad=0,
    half=True
)

img = Image.open('Demo_img.jpg').convert('RGB') # load the image
img = np.array(img) # convert the image to a numpy array    

output, _ = upsampler.enhance(img, outscale=4) # upscale the image

output_img = Image.fromarray(output) # convert the output to an image
output_img.save('output.png') # save the output