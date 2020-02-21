import os
import matplotlib.pyplot as plt
import numpy as np
import stain_utils
import stainNorm_Vahadane
from PIL import Image
import multiprocessing

data_dir = "/projects/shart/digital_pathology/data/biliary/WSI_extraction"
out_dir = "/projects/shart/digital_pathology/data/biliary/Normalized_patches"
case_folder = os.listdir(data_dir)
img_file_name_list = []
cnt = 0
total_cnt = 103407+50651
for cf in case_folder:
    patch_names = os.listdir(os.path.join(data_dir, cf))
    for p_name in patch_names:
        # if cnt % 500 ==0:
        #     print("Processing %d / %d" % (cnt, total_cnt))
        # cnt += 1
        img_file_name = os.path.join(data_dir,cf,p_name)
        img_file_name_list.append(img_file_name)


def mp_normalization(img_file_name):
    print("Processing %s" % img_file_name)
    fp, p_l_name = os.path.split(img_file_name)
    _, cf_l = os.path.split(fp)
    img = stain_utils.read_image(img_file_name)
    out_l_dir = "/projects/shart/digital_pathology/data/biliary/Normalized_patches"
    ref_patch = "/projects/shart/digital_pathology/data/biliary/template/Biliary_Positive_0010_11_54100_9400_0_256_256.jpg"
    img_ref = stain_utils.read_image(ref_patch)
    normalizer = stainNorm_Vahadane.Normalizer()
    normalizer.fit(img_ref)
    img_nor = normalizer.transform(img)
    Img = Image.fromarray(img_nor)
    if not os.path.exists(os.path.join(out_l_dir, cf_l)):
        os.makedirs(os.path.join(out_l_dir, cf_l))
    Img.save(os.path.join(out_l_dir, cf_l, p_l_name))


# cores = multiprocessing.cpu_count()-2
cores = 10
print("There are %d CPU cores,use %d" % (multiprocessing.cpu_count(), cores))
pool = multiprocessing.Pool(processes=cores)
norm_array_images = pool.map(mp_normalization, img_file_name_list)




