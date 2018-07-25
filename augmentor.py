import numpy as np
from scipy import ndimage


import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

import numpy as np
import scipy.io as scio    
import os,re
from PIL import Image
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2

 # Globals
IMAGE_SIZE = 299

root_path = "/home/wonjae/YongHyeok/DataSet/ISIC_TASK3"

def atoi(text) : 
    return int(text) if text.isdigit() else text

def natural_keys(text) :
    return [atoi(c) for c in re.split('(\d+)', text)]

def get_filenames(path):
    filenames = []
    for root, dirnames, filenames in os.walk(path):
        filenames.sort(key = natural_keys)
        rootpath = root
    print(len(filenames))
    return filenames

# NOTE
# Images are assumed to be uint8 0-255 valued.
# For augment function:
#   images shape: (batch_size, height, width, channels=3)
#   labels shape: (batch_size, 3)
def addBlotch(image, max_dims=[0.2,0.2]):
    #add's small black/white box randomly in periphery of image
    new_image = np.copy(image)
    shape = new_image.shape
    max_x = shape[0] * max_dims[0]
    max_y = shape[1] * max_dims[1]
    rand_x = 0
    rand_y = np.random.randint(low=0, high=shape[1])
    rand_bool = np.random.randint(0,2)
    if rand_bool == 0:
        rand_x = np.random.randint(low=0, high=max_x)
    else:
        rand_x = np.random.randint(low=(shape[0]-max_x), high=shape[0])
    size = np.random.randint(low=1, high=7) #size of each side of box
    new_image[rand_x:(size+rand_x), rand_y:(size+rand_y), :] = np.random.randint(0,256)
    return new_image

def shift(image, max_amt=0.2):
    new_img = np.copy(image)
    shape = new_img.shape
    max_x = int(shape[0] * max_amt)
    max_y = int(shape[1] * max_amt)
    x = np.random.randint(low=-max_x, high=max_x)
    y = np.random.randint(low=-max_y, high=max_y)
    return ndimage.interpolation.shift(new_img,shift=[x,y,0])

def addNoise(image, amt=0.005):
    noise_mask = np.random.poisson(image / 255.0 * amt) / amt * 255
    noisy_img = image + (noise_mask)
    return np.array(np.clip(noisy_img, a_min=0., a_max=255.), dtype=np.uint8)

def rotate(image):
    randnum = np.random.randint(1,360)
    new_image = np.copy(image)
    return ndimage.rotate(new_image, angle=randnum, reshape=False)

#randomly manipulates image
#rotate, flip along axis, add blotch, shift
def augment(images, labels=None, amplify=2):
    # INPUT:
    #images shape: (batch_size, height, width, channels=3)
    #labels shape: (batch_size, 3)
    ops = {
        0: addBlotch,
        1: shift,
        2: addNoise,
        3: rotate
    }

    shape = images.shape
    new_images = np.zeros(((amplify*shape[0]), shape[1], shape[2], shape[3]))
    if labels is not None:
        new_labels = np.zeros(((amplify*shape[0]), 7))
    for i in range(images.shape[0]):
        cur_img = np.copy(images[i])
        new_images[i] = cur_img
        if labels is not None:
            new_labels[i] = np.copy(labels[i])
        for j in range(1, amplify):
            add_r = ( j * shape[0] )
            which_op = np.random.randint(low=0, high=4)
            dup_img = np.zeros((1,shape[1], shape[2], shape[3]))
            new_images[i+add_r] = ops[which_op](cur_img)
            if labels is not None:
                new_labels[i+add_r] = np.copy(labels[i])
    if labels is not None:
        return new_images.astype(np.uint8), new_labels.astype(np.uint8)
    else:
        return new_images.astype(np.uint8)

class Aug:
    def __init__(self, MEL_aug_num, BCC_aug_num,AKIEC_aug_num,BKL_aug_num,DF_aug_num,VASC_aug_num):
         # Load each lesion data
        images_MEL=np.load(root_path+'/train_images_task3_299_MEL.npy')
        images_NV=np.load(root_path+'/train_images_task3_299_NV.npy')
        images_BCC=np.load(root_path+'/train_images_task3_299_BCC.npy')
        images_AKIEC=np.load(root_path+'/train_images_task3_299_AKIEC.npy')
        images_BKL=np.load(root_path+'/train_images_task3_299_BKL.npy')
        images_DF=np.load(root_path+'/train_images_task3_299_DF.npy')
        images_VASC=np.load(root_path+'/train_images_task3_299_VASC.npy')

        MEL=np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0])
        NV=np.array([0.0,1.0,0.0,0.0,0.0,0.0,0.0])
        BCC=np.array([0.0,0.0,1.0,0.0,0.0,0.0,0.0])
        AKIEC=np.array([0.0,0.0,0.0,1.0,0.0,0.0,0.0])
        BKL=np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0])
        DF=np.array([0.0,0.0,0.0,0.0,0.0,1.0,0.0])
        VASC=np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0])

        gt_MEL=np.zeros((1113, 7))
        gt_NV=np.zeros((6705, 7))
        gt_BCC=np.zeros((514, 7))
        gt_AKIEC=np.zeros((327, 7))
        gt_BKL=np.zeros((1099, 7))
        gt_DF=np.zeros((115, 7))
        gt_VASC=np.zeros((142, 7))

        for i in range(len(gt_MEL)) :
            gt_MEL[i]=MEL
        for i in range(len(gt_NV)) :
            gt_NV[i]=NV
        for i in range(len(gt_BCC)) :
            gt_BCC[i]=BCC
        for i in range(len(gt_AKIEC)) :
            gt_AKIEC[i]=AKIEC
        for i in range(len(gt_BKL)) :
            gt_BKL[i]=BKL
        for i in range(len(gt_DF)) :
            gt_DF[i]=DF
        for i in range(len(gt_VASC)) :
            gt_VASC[i]=VASC

         # Do augmenting!!
        images_MEL_aug, gt_MEL_aug = augment(images_MEL,gt_MEL,amplify=6)
        images_BCC_aug, gt_BCC_aug = augment(images_BCC,gt_BCC, amplify=13)
        images_AKIEC_aug, gt_AKIEC_aug = augment(images_AKIEC,gt_AKIEC, amplify=20)
        images_BKL_aug, gt_BKL_aug = augment(images_BKL,gt_BKL, amplify=6)
        images_DF_aug, gt_DF_aug = augment(images_DF,gt_DF, amplify=58)
        images_VASC_aug, gt_VASC_aug = augment(images_VASC,gt_VASC, amplify=47)

        
