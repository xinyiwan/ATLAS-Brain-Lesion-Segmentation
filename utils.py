import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
def load_data_MRI(image_path,mask_path):
    
    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    
    images = []
    masks = []

    for image in image_list:
        img = imread(os.path.join(image_path, image), as_gray=True)  # "as_grey"
        img = resize(img, (240, 240), anti_aliasing=True).astype('float32')
        images.append(img)

    for image in image_list:
        mask_img = imread(os.path.join(mask_path, str.replace(image,'.png','_Tumor.png')), as_gray=True)
        mask_img = resize(mask_img, (240, 240), anti_aliasing=True).astype('float32')
        masks.append(mask_img)


    images = np.expand_dims(images, axis = -1)
    masks = np.expand_dims(masks, axis = -1)
    
    return np.array(images), np.array(masks)