from PIL import Image, ImageDraw, ImageChops
import openslide
import cv2
import os
import logging
import numpy as np
import io

from skimage.color import rgb2hsv

threshold = 175
patch_size = (256,256)
step = (256,256)

# Threshold grayscale pixel value
def pixelProcThreshold(intensity):
    if intensity < threshold:
        return 255
    else:
        return 0

'''
rect: [x, y, w, h]
patch_size: []
img_size: [w, h]
'''
# enlarge the detected area a little bit to fit the patch extraction size
def getNearestSamplablePatchfromWSI(rect, patch_size,img_size):
    w_m = rect[2] % patch_size[0]
    h_m = rect[3] % patch_size[1]
    new_rect = np.copy(rect)
    if w_m is not 0:
        new_rect[2] = (rect[2]//patch_size[0] + 1) * patch_size[0]
    if h_m is not 0:
        new_rect[3] = (rect[3]//patch_size[1] + 1) * patch_size[1]
    if (new_rect[0] + new_rect[2]) > img_size[0]:
        new_rect[0] = img_size[0] - new_rect[2]
        if new_rect[0] < 0:
            logging.error("Error! Image width too small")
            return None
    if (new_rect[1] + new_rect[3]) > img_size[1]:
        new_rect[1] = img_size[1] - new_rect[3]
        if new_rect[1] < 0:
            logging.error("Error! Image height too small")
            return None
    return new_rect

###################################################################################################################################

## Following 2 functions do the image classification to remove any uninformative image patches

# saved files with low image content will be discarded
# Consider the image with larger disk size will have more image contents than those with smaller disk size
# Images with larger disk size considered as the informative images
def filter_patch_by_file_size(image_array,file_size_threshold=7.5*1024):
    img = Image.fromarray(image_array,mode="RGB")
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        contents = output.getvalue()
    if len(contents) > file_size_threshold:
        return True
    else:
        return False

# saved files with low image content will be discarded
# Some image patches only have a little bit tissue content on the patch edge, most of its patch area are no tissues there
# Those image patches are considered as the uninformative image patches, which need to be exclude
def filter_by_content_area(rgb_image_array, area_threshold=0.5, brightness=0.85):
    hsv_img = rgb2hsv(rgb_image_array)
    value_img = hsv_img[:, :, 2]
    binary_img = value_img < brightness
    blank_size = np.where(binary_img.flatten() == True)[0].size
    blank_size = blank_size/(rgb_image_array.shape[0]*rgb_image_array.shape[1])
    if blank_size > area_threshold:
        return True
    else:
        return False

#############################################################################################################################

# delete the gold artifacts
def filter_gold_artifacts(rgb_image_array, threshold_area=0.1, gold_H=0.3, gold_V=0.85 ):
    hsv_img = rgb2hsv(rgb_image_array)
    hue_img = hsv_img[:, :, 0]
    binary_img2 = hue_img < gold_H
    value_img = hsv_img[:, :, 2]
    binary_img1 = value_img < gold_V
    binary_img = np.logical_and(binary_img2, binary_img1)
    blank_size = np.where(binary_img.flatten() == True)[0].size
    if blank_size / binary_img.size < threshold_area:
        return True
    else:
        return False

'''
Sample a serials of patches from an image array
img_arr:    image array, type: numpy ndarray
patch_size: height and width, eg. [50,50]
step:       sampling step, eg. [50,50]
area_range: define the sampling start and end with where, [start_x_cor,end_x_cor,start_y_cor,end_y_cor],
            eg. [10,810,100,900]
# Reference: sklearn.feature_extraction.image.extract_patches_2d()            
'''
def slide_img_sampling(img_arr,patch_size,step,area_range=None):
    '''

    :param img_arr:
    :param patch_size:
    :param step:
    :param area_range:
    :return:
    '''
    img_size = img_arr.shape
    if area_range is None:
        w_min = 0
        w_max = img_size[1]
        h_min = 0
        h_max = img_size[0]
    else:
        w_min = area_range[0]
        w_max = area_range[1]
        h_min = area_range[2]
        h_max = area_range[3]
    if w_min<0 | h_min<0 | w_max> img_size[1]| h_max> img_size[0]:
        print("area_range parameters error.")
        return False
    nd_array_patches = np.ndarray([])
    offsets = []
    row_num = 0
    for h in range(h_min, h_max-patch_size[0]+step[0], step[0]):
        for w in range(w_min, w_max-patch_size[1]+step[1], step[1]):
            offsets.append([w,h])
            patch_arr = img_arr[h:(h + patch_size[0]), w:(w + patch_size[1]), :]
            feature = patch_arr.flatten()
            if row_num == 0:
                nd_array_patches = feature
            else:
                nd_array_patches = np.vstack((nd_array_patches, feature))
            row_num += 1
    return nd_array_patches, offsets








