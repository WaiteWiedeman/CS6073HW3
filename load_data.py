# imports
import os
import numpy as np
from PIL import Image


# function to load data
# takes paths as input and returns images and masks
def LoadData(path1, path2):
    # read images and masks from path
    image_dataset = os.listdir(path1)
    mask_dataset = os.listdir(path2)

    # make a list for images and masks filenames
    orig_img = []
    mask_img = []
    for file in image_dataset:
        orig_img.append(file)
    for file in mask_dataset:
        mask_img.append(file)

    # sort the lists
    orig_img.sort()
    mask_img.sort()

    return orig_img, mask_img


# function to preprocess images
# takes images, masks, desired image and mask shapes, and file paths as input
# returns NumPy dataset with preprocessed images as 3-D arrays of desired size
def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2):
    # pull the relevant dimensions for image and mask
    m = len(img)  # number of images
    i_h, i_w, i_c = target_shape_img  # pull height, width, and channels of image
    m_h, m_w, m_c = target_shape_mask  # pull height, width, and channels of mask

    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m, i_h, i_w, i_c), dtype=np.float32)
    y = np.zeros((m, m_h, m_w, m_c), dtype=np.int32)

    # Resize images and masks
    for file in img:
        # convert image into an array of desired shape (3 channels)
        index = img.index(file)
        path = os.path.join(path1, file)
        single_img = Image.open(path).convert('RGB')
        single_img = single_img.resize((i_h, i_w))
        single_img = np.reshape(single_img, (i_h, i_w, i_c))
        single_img = single_img  # / 256.0
        X[index] = single_img

        # convert mask into an array of desired shape (1 channel)
        single_mask_ind = mask[index]
        path = os.path.join(path2, single_mask_ind)
        single_mask = Image.open(path)
        single_mask = single_mask.resize((m_h, m_w))
        single_mask = np.array(single_mask)
        single_mask = np.reshape(single_mask, (m_h, m_w, m_c))
        single_mask[single_mask <= 35] = 0
        single_mask[single_mask > 35] = 1
        y[index] = single_mask
    return X, y


