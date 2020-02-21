from PIL import Image, ImageDraw, ImageChops
import openslide
import cv2
import os
from skimage import measure
import numpy as np
import argparse
# import stain_utils
from WSI_extraction_utils import filter_patch_by_file_size, pixelProcThreshold, filter_by_content_area, filter_gold_artifacts, getNearestSamplablePatchfromWSI,slide_img_sampling


parser = argparse.ArgumentParser(description='create metadata from text file, including complaints and ICD codes, for disease prediction')
parser.add_argument("-i", "--data_dir", dest='data_dir', default="//mfad.mfroot.org/researchmn/DLMP-MACHINE-LEARNING/BiliaryCytology/req19175_1-22-2019", type=str,required=False, help="Image directory")
parser.add_argument("-n", "--wsi_name", dest='wsi_name', default="", type=str, required=True,help="WSI name")
parser.add_argument("-o", "--data_out_dir", dest='data_out_dir', default="H:\\BiliaryStains\\WSI_extraction", type=str, required=False,help="Image patch output directory")

args = parser.parse_args()

data_dir = args.data_dir
wsi_name = args.wsi_name
data_out_dir = args.data_out_dir

threshold = 175  # Threshold for tissue detection
patch_size = (256, 256)  # define patch size for patch extraction
step = (256, 256)   # stride for patch extraction
im_channel = 3   # number of image channel

print("Processing %s" % wsi_name)
slide_path = data_dir + "/" + wsi_name
# read WSI thumbnail
if not os.path.exists(data_out_dir):
    os.makedirs(data_out_dir)
WSI = openslide.OpenSlide(slide_path)
WSI_Width, WSI_Height = WSI.dimensions
thumb_size_x = round(WSI_Width / 100)
thumb_size_y = round(WSI_Height / 100)
thumbnail = WSI.get_thumbnail([thumb_size_x, thumb_size_y])
thumbnail = thumbnail.convert('L')  # get grayscale image
BinaryImg = thumbnail.point(pixelProcThreshold)  # get threshold image
# thumbnail.save(os.path.join(data_out_dir, wsi_name[0:-4] + ".jpg"))
# BinaryImg.save(os.path.join(data_out_dir, wsi_name[0:-4] + "_thr_" + str(threshold) + ".jpg"))
b_img = np.array(BinaryImg)
blobs_labels = measure.label(b_img, background=0)
# locate bounding boxes
area_min = 4
box_padding = 4
(Img_w, Img_h) = b_img.shape
labels = range(blobs_labels.min(), blobs_labels.max())  # how many labels in this binary image
rect_list = []
for l in labels[1:]:  # ignore background
    mask_l = blobs_labels[blobs_labels == l]
    if mask_l.size > area_min:  # Keep the area of the connected components larger than 100 pixels
        mask = blobs_labels == l  # select pixels for label l
        mask = np.array(mask, dtype=np.uint8)
        x, y, w, h = cv2.boundingRect(mask)
        if w < 22 and h < 22:  # limit the size of the bounding box, if the size of the bounding box is too large, that might be artifacts
            # Note! TF-Record save the bounding box as [top_left_cor_x,top_left_cor_x,bottom_left_cor_x,bottom_left_cor_y]
            rect_list.append([x, y, w, h])
            #print("add to list, mask size: %d" % mask_l.size)
print("\t %d targets in this image" % rect_list.__len__())
# validateBoundingBox
draw = ImageDraw.Draw(thumbnail)
for b in rect_list:
    xy = [b[0], b[1], b[0]+b[2], b[1]+b[3]]
    draw.rectangle(xy, outline='red')
del draw
# thumbnail.save(os.path.join(data_out_dir, wsi_name[0:-4] + "_thr_" + str(threshold) + "_boxes.jpg"))
# zoom in to level 0 at the detected locations
patch_dir = os.path.join(data_out_dir, wsi_name[0:-4]+"_patches")
patch_dir = patch_dir.replace(" ", "_")
if not os.path.exists(patch_dir):
    os.makedirs(patch_dir)
print("Extracting useful image patches from target area")
for b_idx, b in enumerate(rect_list):
    xy = [b[0]*100, b[1]*100, b[2]*100, b[3]*100]
    # resize the bounding box to get a sample-able patch.
    new_rect = getNearestSamplablePatchfromWSI(xy, patch_size, [WSI_Width, WSI_Height])
    BoxPatch = WSI.read_region([new_rect[0], new_rect[1]], 0, [new_rect[2], new_rect[3]]).convert("RGB")
    # BoxPatch.save(os.path.join(patch_dir, wsi_name[0:-4] + "_" + str(b_idx) + ".jpg"))
    img_arr = np.array(BoxPatch)
    ImgPatches, offsets = slide_img_sampling(img_arr, patch_size, step)
    for p_idx, p in enumerate(ImgPatches):
        [w, h] = offsets[p_idx]
        patch = np.reshape(p, [patch_size[0], patch_size[1], im_channel])
        OutName = "_".join((wsi_name[0:-4], str(b_idx), str(new_rect[0] + w), str(new_rect[1] + h),str(patch_size[0]), str(patch_size[1])))
        OutName = OutName.replace(" ", "_")
        # if filter_patch_by_file_size(patch):   # filter by the size of jepg encoding size
        if filter_by_content_area(patch):  # filter by the content area
            if filter_gold_artifacts(patch):  # filter the gold artifacts
                patch = Image.fromarray(patch)
                # img_nor = patch_normalization(patch, img_ref)
                # patch = Image.fromarray(img_nor)
                patch.save(os.path.join(patch_dir, OutName + ".jpg"))
        else:
            # patch = Image.fromarray(patch)
            # rej_path = patch_dir.replace("_patches", "_rejects")
            # if not os.path.exists(rej_path):
            #     os.makedirs(rej_path)
            # patch.save(os.path.join(rej_path, OutName + ".jpg"))
            print("Discard image:%s" % OutName)














