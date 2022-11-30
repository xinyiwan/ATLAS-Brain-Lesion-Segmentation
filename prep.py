import os
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


images_list = glob.glob("/home/k8s-group5/Training/*/*/*/*/*T1w.nii.gz")

working_path = '/home/k8s-group5'
images_path = os.path.join(working_path,"Images_cv")
masks_path = os.path.join(working_path,"Masks_cv")
if not os.path.exists(images_path):
    os.mkdir(images_path)
if not os.path.exists(masks_path):
    os.mkdir(masks_path)

for i in images_list:

    label = i.split("/anat/")[1].split("_ses")[0]

    i_mask = i.replace("_T1w.nii.gz","_label-L_desc-T1lesion_mask.nii.gz")
    mask = nib.load(i_mask).get_fdata()
    img = nib.load(i).get_fdata()


    for idx in range(50,mask.shape[-1]-50):
        if 1 in mask[:,:,idx]:

            # Save masks as binary img
            mask_v = mask[:,:,idx] * 255
            mask_file = label + "_{}_".format(idx) + "mask.png"
            cv.imwrite(os.path.join(masks_path,mask_file), mask_v)

            # plt.imshow(mask[:,:,idx],cmap = 'gray')
            # plt.axis('off')
            # plt.savefig(os.path.join(masks_path,mask_file), bbox_inches = 'tight', pad_inches = 0)


            # Save imgs 
            img_file = label + "_{}_".format(idx) + "img.png"
            cv.imwrite(os.path.join(images_path,img_file),img[:,:,idx])

            # plt.imshow(img[:,:,idx],cmap = 'gray')
            # plt.axis('off')
            # plt.savefig(os.path.join(images_path,img_file),  bbox_inches = 'tight', pad_inches = 0)