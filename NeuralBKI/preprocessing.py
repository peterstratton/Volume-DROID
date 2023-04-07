#Preprocessing script for converting .npy files into images
import numpy as np
from PIL import Image

file_name = '000173_left_depth.npy' #Insert file name
img_array = np.load(file_name)

#To visualize depth .npy files, increasing images by factor of 20
img_array *= 20

# print(img_array[100:150,100:150])

#Normalizing the depth image -- NOTE: This might leave the image darker than a simple multiplier
mn = np.min(img_array)
mx = np.max(img_array)
img_array = (img_array - mn) * (1.0 / (mx - mn)) * 255

im = Image.fromarray(img_array)

if im.mode != 'RGB':
    im = im.convert('RGB')

im.save("sample.jpg")

