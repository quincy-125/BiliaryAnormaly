import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time
from data_preparation_utils import get_subimgvec_by_imglist, split_PN_samples
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

def get_closest_dists(compare_nodes, reference_nodes):
    dist_matrix = pairwise_distances(reference_nodes.astype(np.float32), compare_nodes.astype(np.float32))
    # print(dist_matrix.shape)
    min_dists = np.amin(dist_matrix, axis=0)
    # print(min_dists)
    return min_dists

# find the closest node with current node and a node list
def find_closest_nodes(compare_nodes, reference_nodes):
    min_dist_list = []
    min_dist_idx_list = []
    for node in compare_nodes:
        node_np_array = np.asarray(node).astype(np.float32)
        deltas = reference_nodes.astype(np.float32) - node_np_array
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        min_dist_idx = np.argmin(dist_2)
        min_dist_idx_list.append(min_dist_idx)
        min_dist_list.append(dist_2[min_dist_idx])
    return min_dist_list, min_dist_idx_list

# usage: create_weight_map("original.jpg",patch_weights,"weights.jpg")
def create_weight_map(image_name, patch_weights, save_name, colormap='jet',sub_split=(8,8)):
    fig, ax = plt.subplots(1,2,figsize=(6,3))
    I = Image.open(image_name)
    ax[0].imshow(np.array(I))
    ax[0].axis('off')
    plt.title("min:"+str(min(patch_weights))+",max:"+str(max(patch_weights)))
    data = np.array(patch_weights).reshape(sub_split)
    ax[1].matshow(data, cmap=colormap)
    ax[1].axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.8)
    plt.savefig(save_name)

def update_min_max(dist_list, global_min_dist,global_max_dist):
    local_min_dist = min(dist_list)
    local_max_dist = max(dist_list)
    if local_min_dist > global_min_dist:
        global_min_dist = local_min_dist
    if local_max_dist < global_max_dist:
        global_max_dist = local_max_dist
    return global_min_dist, global_max_dist

t0 = time.time()
stride = 64
data_dir = "/projects/shart/digital_pathology/data/biliary/EncodeImgs"
model_save_dir = "/projects/shart/digital_pathology/data/biliary/models/kmeans"
model_eval_dir = "/projects/shart/digital_pathology/data/biliary/models_eval/kmeans"
# load image names for indexing negative and positive samples
image_names_file = os.path.join(data_dir, "Content_rich_artifacts_free_filenames.npy")
image_names = np.load(image_names_file)
pos_idx, pos_image_names, neg_idx, neg_image_names = split_PN_samples(image_names,indicator=('Positive','Negative'))

# load image representation from file
img_vec_file = os.path.join(data_dir, "Small_norm_patch_codes_8.npy")
img_vectors = np.load(img_vec_file)["codes"]
print(img_vectors.shape)

# select negative ones
Img_vec_neg = np.zeros(shape=(len(neg_idx)*stride,img_vectors.shape[1]))
for idx,n_idx in enumerate(neg_idx):
    sub_patch_vectors = img_vectors[n_idx*stride:(n_idx+1)*stride, :]
    Img_vec_neg[idx*stride:(idx+1)*stride, :] = sub_patch_vectors
print(Img_vec_neg.shape)

# select positive ones
Img_vec_pos = np.zeros(shape=(len(pos_idx)*stride,img_vectors.shape[1]))
for idx,n_idx in enumerate(pos_idx):
    sub_patch_vectors = img_vectors[n_idx*stride:(n_idx+1)*stride, :]
    Img_vec_pos[idx*stride:(idx+1)*stride, :] = sub_patch_vectors
print(Img_vec_pos.shape)


# load images to be represented
# Test negative images' representation
# cnt = 0
neg_rep_size = int(len(neg_image_names)*0.8)
neg_rep_vec = Img_vec_neg[0:neg_rep_size*stride]
neg_test_vec = Img_vec_neg[neg_rep_size*stride:]
neg_test_img_names = neg_image_names[neg_rep_size:]
# for idx, n_img_name in enumerate(neg_test_img_names):
#     sub_patch_vectors = neg_test_vec[idx*stride:(idx+1)*stride, :]
#     # for each sub-patches in a testing negative image, find the most similar sub-patch from all negative sub-patches
#     min_dist_list, min_dist_idx_list = find_closest_nodes(sub_patch_vectors, neg_rep_vec)
#     print(n_img_name)
#     print(min_dist_list)
#     f_name = os.path.split(n_img_name)[1]
#     save_file_name = os.path.join(model_eval_dir, "weights-"+f_name+"-.jpg")
#     create_weight_map(n_img_name, min_dist_list, save_file_name)
#     cnt += 1
#     if cnt > 10:
#         break

# # Positive patches representation
# cnt = 0
# for idx, p_img_name in enumerate(pos_image_names):
#     sub_patch_vectors, image_idx = get_subimgvec_by_imglist(img_vectors, p_img_name, list(image_names), stride=64)
#     # for each sub-patches in a positive image, find the most similar sub-patch from all negative sub-patches
#     min_dist_list, min_dist_idx_list = find_closest_nodes(sub_patch_vectors, Img_vec_neg)
#     print(p_img_name)
#     print(min_dist_list)
#     f_name = os.path.split(p_img_name)[1]
#     save_file_name = os.path.join(model_eval_dir,"weights-"+f_name+"-.jpg")
#     create_weight_map(p_img_name,min_dist_list,save_file_name)
#     cnt += 1
#     if cnt > 10:
#         break

# positive distribution
all_pos_dist = np.zeros(shape=(len(pos_idx), stride))
global_min_dist = 0.0
global_max_dist = 9999999999999999999999.0
dists_sz = len(pos_idx)
# for idx in range(dists_sz):
#     if idx % 100 == 0:
#         print("Processing positive %d / %d" % (idx, dists_sz))
#     pos_vec = Img_vec_pos[idx*stride:(idx+1)*stride, :]
#     # min_dist_list, min_dist_idx_list = find_closest_nodes(pos_vec, Img_vec_neg)
#     # min_dist_list = get_closest_dists(pos_vec, Img_vec_neg)
#     min_dist_list = get_closest_dists(pos_vec, neg_rep_vec)
#     all_pos_dist[idx,:] = min_dist_list
#     # print(all_pos_dist[idx,:])
# print(all_pos_dist.shape)
save_all_pos_dist_file = os.path.join(model_eval_dir, "all_pos_dist.npy")
# np.save(save_all_pos_dist_file, all_pos_dist)
# #
# # # # negative distribution
test_neg_dist = np.zeros(shape=(len(neg_test_img_names), stride))
dists_sz_test_neg = len(neg_test_img_names)
for idx in range(dists_sz_test_neg):
    if idx % 100 == 0:
        print("Processing negative %d / %d" % (idx, dists_sz_test_neg))
    neg_vec = neg_test_vec[idx*stride:(idx+1)*stride, :]
    # min_dist_list, min_dist_idx_list = find_closest_nodes(neg_vec, neg_rep_vec)
    min_dist_list = get_closest_dists(neg_vec, neg_rep_vec)
    test_neg_dist[idx,:] = min_dist_list
    # print(all_pos_dist[idx,:])
print(test_neg_dist.shape)
save_test_pos_dist_file = os.path.join(model_eval_dir, "dists_sz_test_neg.npy")
# np.save(save_test_pos_dist_file, test_neg_dist)
#
# print("Plotting distribution")

all_pos_dist = np.load(save_all_pos_dist_file)
test_pos_dist = np.load(save_test_pos_dist_file)
bins_no = int(max(all_pos_dist.flatten()))

plt.figure(1)
plt.hist([all_pos_dist.flatten(), test_neg_dist.flatten()], bins=bins_no, density=True, color=['orange', 'blue'], alpha=0.75, linewidth=2, histtype='step')
plt.grid()
plt.legend(["Negative","Positive"], loc='upper center')# be aware! reversed legend.
# Explaination: https://stackoverflow.com/questions/47084606/column-order-reversed-in-step-histogram-plot
plt.title("Distance distribution")
plt.savefig(os.path.join(model_eval_dir, "dist_distribution.jpg"))

plt.figure(2)
plt.hist(all_pos_dist.flatten(), bins=bins_no, density=True, color='orange', alpha=0.75, linewidth=2, histtype='step')
plt.grid()
plt.legend(["Positive"], loc='upper center')
plt.title("Positive distance distribution")
plt.savefig(os.path.join(model_eval_dir, "pos_distribution.jpg"))

plt.figure(3)
plt.hist(test_neg_dist.flatten(), bins=bins_no, density=True, color='blue', alpha=0.75, linewidth=2, histtype='step')
plt.grid()
plt.legend(["Negative"], loc='upper center')
plt.title("Negative distance distribution")
plt.savefig(os.path.join(model_eval_dir, "neg_distribution.jpg"))


plt.figure(4)
plt.hist(all_pos_dist.flatten(), bins="auto", facecolor='orange', alpha=0.75)
plt.grid()
plt.legend(["Positive"], loc='upper center')
plt.title("Positive distance distribution")
plt.savefig(os.path.join(model_eval_dir, "pos_auto_distribution.jpg"))



'''
Globally normalized hot map
'''
# min_dist_list_array = []
# global_min_dist = 0.0
# global_max_dist = 9999999999999999999999.0
# for idx, p_img_name in enumerate(pos_image_names):
#     sub_patch_vectors, image_idx = get_subimgvec_by_imglist(img_vectors, p_img_name, list(image_names), stride=64)
#     # for each sub-patches in a positive image, find the most similar sub-patch from all negative sub-patches
#     min_dist_list, min_dist_idx_list = find_closest_nodes(sub_patch_vectors, Img_vec_neg)
#     global_min_dist, global_max_dist = update_min_max(min_dist_list, global_min_dist,global_max_dist)
#     min_dist_list_array.append(min_dist_list)
#
# cnt = 0
# for idx, p_img_name in enumerate(pos_image_names):
#     f_name = os.path.split(p_img_name)[1]
#     print(p_img_name)
#     sub_patch_dist_list = min_dist_list_array[idx]
#     scale = (global_max_dist - global_min_dist) / (max(sub_patch_dist_list) - min(sub_patch_dist_list))
#     normal_min_dist_list_array = np.asarray(sub_patch_dist_list)*scale
#     print(normal_min_dist_list_array)
#
#     save_file_name = os.path.join(model_eval_dir, "weights-"+f_name+"-.jpg")
#     create_weight_map(p_img_name, normal_min_dist_list_array, save_file_name)
#     cnt += 1
#     if cnt > 10:
#         break


t1 = time.time()
print("Spend %s s" % str(t1-t0))
