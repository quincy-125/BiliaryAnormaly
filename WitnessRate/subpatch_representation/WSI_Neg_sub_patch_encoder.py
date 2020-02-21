import numpy as np
import os
from batch_encoder import batch_encoder, batch_arr_encoder


SAVE = True
ITER = 8
ImgSize = (32, 32, 3)

# encode the image sub-patches into compressed code
file_list = ["NegTrainingSubPatches_r0.npy", "NegTrainingSubPatches_r90.npy", "NegTrainingSubPatches_r180.npy","NegTrainingSubPatches_r270.npy"]
data_dir = "/projects/shart/digital_pathology/data/biliary/WSI_Neg_encode"

for f in file_list:
    small_norm_patches_npy = os.path.join(data_dir, f)
    path_ele = os.path.split(small_norm_patches_npy)
    encoded_patch_save_name = os.path.join(path_ele[0], "Encoded_"+path_ele[1][0:-4]+"_it"+str(ITER)+".npy")
    small_norm_array_images = np.load(small_norm_patches_npy)
    batch_arr_encoder(small_norm_array_images, ImgSize, encoded_patch_save_name, SAVE, iteration=ITER)











