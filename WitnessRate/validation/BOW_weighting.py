import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import collections
import operator
from scipy.signal import savgol_filter

data_root_dir = "/projects/shart/digital_pathology/data/biliary/models_eval/BOW"
eval_out_dir = "/projects/shart/digital_pathology/data/biliary/models_eval/BOW"

# neg_mis_npy_dir = os.path.join(data_root_dir, "Neg_direct_dist")
# pos_mis_npy_dir = os.path.join(data_root_dir, "Pos_direct_dist")

# Function returns N largest elements
def Nmaxelements(list1, N):
    final_list = []
    for i in range(0, N):
        max1 = 0
        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j]
        list1.remove(max1)
        final_list.append(max1)
    return final_list

neg_ass_npy_dir = os.path.join(data_root_dir, "Neg_KM_ass")
neg_npy_files = os.listdir(neg_ass_npy_dir)
neg_assignments = []
for nf in neg_npy_files:
    neg_ass = np.load(os.path.join(neg_ass_npy_dir, nf))
    neg_assignments.append(list(neg_ass.flatten()))
neg_patch_cnt = len(neg_assignments)

neg_assignments_flatten = list(itertools.chain(*neg_assignments))
neg_sub_patches_cnt = len(neg_assignments_flatten)

std_val = np.std(neg_assignments_flatten)
avr_val = np.average(neg_assignments_flatten)
print("Negative patch_cnt: %d sub-patch_cnt: %d " % (neg_patch_cnt, neg_sub_patches_cnt))
print("Negative Average: %.4f, STD: %.4f" % (avr_val, std_val))
###########################################
# get weight for clusters
cluster_weights = {}
for k in neg_assignments_flatten:
    cluster_weights[k] = cluster_weights.get(k, 0) + 1
for k in cluster_weights.keys():
    cluster_weights[k] = 1.0 / cluster_weights.get(k, 0)
###########################################
# test positive patches
pos_ass_npy_dir = os.path.join(data_root_dir, "Pos_KM_ass")
pos_npy_files = os.listdir(pos_ass_npy_dir)
pos_assignments = []
for pf in pos_npy_files:
    pos_ass = np.load(os.path.join(pos_ass_npy_dir, pf))
    pos_assignments.append(list(pos_ass.flatten()))

pos_patch_cnt = len(pos_assignments)
pos_assignments_flatten = list(itertools.chain(*pos_assignments))
pos_sub_patches_cnt = len(pos_assignments_flatten)

std_val = np.std(pos_assignments_flatten)
avr_val = np.average(pos_assignments_flatten)
print("Positive patch_cnt: %d sub-patch_cnt: %d " % (pos_patch_cnt, pos_sub_patches_cnt))
print("Positive Average: %.4f, STD: %.4f" % (avr_val, std_val))
# use all the sub-patches
# pos_patch_scores = []
# for p_ass in pos_assignments:
#     patch_score = 0
#     for sub_ass in p_ass:
#         patch_score += cluster_weights.get(sub_ass, 0)
#     pos_patch_scores.append(patch_score)
neg_patch_scores = []
for n_ass in neg_assignments:
    patch_score = []
    for sub_ass in n_ass:
        patch_score.append(cluster_weights.get(sub_ass, 0))
    top_scores1 = max(patch_score)
    patch_score.remove(top_scores1)
    top_scores2 = max(patch_score)
    neg_patch_scores.append(top_scores1 + top_scores2)

# use only top 2
pos_patch_scores = []
for p_ass in pos_assignments:
    patch_score = []
    for sub_ass in p_ass:
        patch_score.append(cluster_weights.get(sub_ass, 0))
    top_scores1 = max(patch_score)
    patch_score.remove(top_scores1)
    top_scores2 = max(patch_score)
    pos_patch_scores.append(top_scores1+top_scores2)

x = np.arange(0, len(pos_patch_scores), 1)
plt.figure(dpi=300)
# plt.plot(x, pos_patch_scores)
bins_no = int(len(set(pos_patch_scores))/2)
plt.hist([pos_patch_scores, neg_patch_scores], bins=bins_no, color=['orange','blue'], alpha=0.75)
plt.legend(["P", "N"])
plt.title('P%d, N%d' % (pos_sub_patches_cnt, neg_sub_patches_cnt))
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.grid()

plt.savefig(os.path.join(eval_out_dir,"PN_sample_sore.png"))
###########################################


















