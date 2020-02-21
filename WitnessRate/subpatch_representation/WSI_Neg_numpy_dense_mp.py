# save all the negative samples into
import os
import random
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
from skimage.color import rgb2hsv
import platform
import multiprocessing

Rotation = 90
step = [32, 32]
# step = [16, 16]

ImgSize = [256, 256]
patch_size = [32, 32]
img_channels = 3
blank_V_threshold = 0.85

print(platform.platform())
if "Windows" in platform.platform():
    data_root_dir = "H:\\BiliaryStains\\threshold"
    NegTrainingSubPatches_npy_dir = "H:\\BiliaryStains"

    training_list_np_file = "H:\\BiliaryStains\\neg_training_list.npy"
    testing_list_np_file = "H:\\BiliaryStains\\neg_testing_list.npy"
else:
    data_root_dir = "/projects/shart/digital_pathology/data/biliary/Normalized_patches"
    NegTrainingSubPatches_npy_dir = "/projects/shart/digital_pathology/data/biliary/WSI_Neg_encode/CaseSubPatches"

    training_list_np_file = "/projects/shart/digital_pathology/data/biliary/WSI_Neg_encode/neg_training_list.npy"
    testing_list_np_file = "/projects/shart/digital_pathology/data/biliary/WSI_Neg_encode/neg_testing_list.npy"

if not os.path.exists(training_list_np_file):
    total_neg_case = 119
    exclude_neg = 75

    neg_case_list = [i for i in range(1, total_neg_case+1)]
    neg_case_list.remove(exclude_neg)
    random.shuffle(neg_case_list)

    training_size = int(len(neg_case_list)*0.8) #  80% as training set
    training_sample_list = random.sample(neg_case_list,training_size)
    train_case_names = []
    test_case_names = []
    for s in neg_case_list:
        if s in training_sample_list:
            train_case_names.append("Biliary_Negative_{0:04d}_patches".format(s))
        else:
            test_case_names.append("Biliary_Negative_{0:04d}_patches".format(s))
    np.save(training_list_np_file, train_case_names)
    np.save(testing_list_np_file, test_case_names)
else:
    train_case_names = np.load(training_list_np_file)
    test_case_names = np.load(testing_list_np_file)

####################################
def slide_img_sampling(img_arr,patch_size,step,area_range=None):
    '''
    :param img_arr:
    :param patch_size:
    :param step:
    :param area_range: rect which defines the sampling area
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
    nd_array_patches = np.ndarray([], dtype=np.uint8)
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

####################################
def slide_img_sampling_with_mask(img_arr,patch_size,step,mask_arr, max_sample_num=136):
    '''
    :param img_arr:
    :param patch_size:
    :param step:
    :param mask_arr:
    :return:
    '''
    nd_array_patches = np.ndarray([], dtype=np.uint8)
    i_w = img_arr.shape[0]
    i_h = img_arr.shape[1]
    row_num = 0
    for h in range(0, i_h-patch_size[0]+step[0], step[0]):
        for w in range(0, i_w-patch_size[1]+step[1], step[1]):
            if mask_arr[h, w]:
                patch_arr = img_arr[h:(h + patch_size[0]), w:(w + patch_size[1]), :]
                feature = patch_arr.flatten()
                if row_num == 0:
                    nd_array_patches = feature
                else:
                    nd_array_patches = np.vstack((nd_array_patches, feature))
                row_num += 1
    if len(nd_array_patches) > max_sample_num:
        randomRow = np.random.randint(len(nd_array_patches), size=max_sample_num)
        return nd_array_patches[randomRow,:]
    else:
        return nd_array_patches

#########################################



# test function stain_patch_sampling
# image_name = "H:\\BiliaryStains\\threshold\Biliary_Positive_0003_patches\\Biliary_Positive_0003_20_32756_8056_0_256_256.jpg"
# test_dir = "H:\\BiliaryStains\\patch_filter_test\\sub_patches"
# gold_V = 0.8
# RBG_img = np.array(Image.open(image_name, 'r'))
# HSV_img = rgb2hsv(RBG_img)
# value_img = HSV_img[:, :, 2]
# mask_img = np.vstack((value_img < gold_V))
# patches, _ = slide_img_sampling_with_mask(RBG_img, patch_size, [8,8], mask_img)
# for idx, p in enumerate(patches):
#     img = Image.fromarray(p.reshape([patch_size[0], patch_size[1], img_channels]))
#     img_n = os.path.join(test_dir, str(idx)+".jpg")
#     img.save(img_n)

#######################


training_img_names = []
for cn in train_case_names:
    case_dir = os.path.join(data_root_dir, cn)
    image_names = os.listdir(case_dir)
    for iname in image_names:
        img_fn = os.path.join(case_dir, iname)
        training_img_names.append(img_fn)

def extract_sub_patches_by_case(case_name):
    print("Processing case: %s" % case_name)
    save_name_npy = os.path.join(NegTrainingSubPatches_npy_dir, case_name + "_r" + str(Rotation) + ".npy")
    if not os.path.exists(save_name_npy):
        case_dir = os.path.join(data_root_dir, case_name)
        image_names = os.listdir(case_dir)
        sample_dim = patch_size[0] * patch_size[1] * img_channels
        training_sub_patches = np.empty([0, sample_dim])
        for iname in image_names:
            img_name = os.path.join(case_dir, iname)
            Img = Image.open(img_name, "r")
            if Rotation > 0:
                Img.rotate(Rotation)
            RBG_img = np.array(Img)
            HSV_img = rgb2hsv(RBG_img)
            value_img = HSV_img[:, :, 2]
            mask_img = np.vstack((value_img < blank_V_threshold))
            patches = slide_img_sampling_with_mask(RBG_img, patch_size, step, mask_img)
            # if sample_idx % 10 == 0:
            print("Get %d content rich sub-patches" % len(patches))
            training_sub_patches = np.vstack((training_sub_patches, patches))
        # save_name_npy = os.path.join(NegTrainingSubPatches_npy_dir, case_name+"_r"+str(Rotation)+".npy")
        np.save(save_name_npy, training_sub_patches)


# cores = multiprocessing.cpu_count()-2
cores = 4
print("There are %d CPU cores,use %d" % (multiprocessing.cpu_count(), cores))
pool = multiprocessing.Pool(processes=cores)
norm_array_images = pool.map(extract_sub_patches_by_case, train_case_names)

















