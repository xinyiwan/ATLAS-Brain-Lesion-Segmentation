from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import random
import cv2 as cv

def load_data(img_w, img_h, image_path,mask_path):
    
    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    s = list(zip(image_list,mask_list))
    random.shuffle(s)
    image_list, mask_list = zip(*s)
    
    images = []
    masks = []
    idx_i = 0 
    idx_m = 0
    w, h = 233, 233

    ## Read images
    for image in image_list[0:8000]:
        img = imread(os.path.join(image_path, image), as_gray=True)  # "as_grey"
        img_data = np.zeros((w, h))
        img_data[18:215,:] = img
        img = resize(img_data, (img_w, img_h), anti_aliasing=True).astype('float32')
        images.append(img)
        idx_i += 1
        if idx_i % 100 == 0:
             print('Reading: {0}/{1}  of train images'.format(idx_i, len(image_list)))

    ## Read masks
    for image in image_list[0:8000]:
        m_path = os.path.join(mask_path, str.replace(image,'img','mask'))
        mask_img = np.array(cv.imread(m_path, 0), dtype=np.uint8)
        mask_img = np.divide(mask_img, 255)

        mask_data = np.zeros((w, h))
        mask_data[18:215,:] = mask_img
        mask_img = resize(mask_data, (img_w, img_h), anti_aliasing=True).astype('float32')
        masks.append(mask_img)
        idx_m += 1
        if idx_m % 100 == 0:
             print('Reading: {0}/{1}  of train masks'.format(idx_m, len(image_list)))


    images = np.expand_dims(images, axis = -1)
    masks = np.expand_dims(masks, axis = -1)
    
    return np.array(images), np.array(masks)

def load_data_MRI(image_path,mask_path):
    
    image_list = os.listdir(image_path)
    
    images = []
    masks = []
    idx_i = 0 
    idx_m = 0

    ## Read images
    for image in image_list[0:100]:
        img = imread(os.path.join(image_path, image), as_gray=True)  # "as_grey"
        img = resize(img, (240, 240), anti_aliasing=True).astype('float32')
        images.append(img)
        idx_i += 1
        if idx_i % 10 == 0:
             print('Reading: {0}/{1}  of train images'.format(idx_i, len(image_list[0:100])))

    ## Read masks
    for image in image_list[0:100]:
        mask_img = imread(os.path.join(mask_path, str.replace(image,'.png','_Tumor.png')), as_gray=True)
        mask_img = resize(mask_img, (240, 240), anti_aliasing=True).astype('float32')
        masks.append(mask_img)
        idx_m += 1
        if idx_m % 10 == 0:
             print('Reading: {0}/{1}  of train masks'.format(idx_m, len(image_list[0:100])))


    images = np.expand_dims(images, axis = -1)
    masks = np.expand_dims(masks, axis = -1)
    
    return np.array(images), np.array(masks)