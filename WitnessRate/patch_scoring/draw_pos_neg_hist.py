import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import collections
import operator
from scipy.signal import savgol_filter

data_root_dir = "/projects/shart/digital_pathology/data/biliary/models_eval/BOW"

# neg_mis_npy_dir = os.path.join(data_root_dir, "Neg_direct_dist")
# pos_mis_npy_dir = os.path.join(data_root_dir, "Pos_direct_dist")

neg_mis_npy_dir = os.path.join(data_root_dir, "Neg_KM_ass")
pos_mis_npy_dir = os.path.join(data_root_dir, "Pos_KM_ass")


neg_npy_files = os.listdir(neg_mis_npy_dir)
pos_npy_files = os.listdir(pos_mis_npy_dir)

neg_distances = []
pos_distances = []
for nf in neg_npy_files:
    neg_dist = np.load(os.path.join(neg_mis_npy_dir, nf))
    neg_distances.append(list(neg_dist.flatten()))
for pf in pos_npy_files:
    pos_dist = np.load(os.path.join(pos_mis_npy_dir, pf))
    pos_distances.append(list(pos_dist.flatten()))

neg_patch_cnt = len(neg_distances)
pos_patch_cnt = len(pos_distances)

neg_distances = list(itertools.chain(*neg_distances))
pos_distances = list(itertools.chain(*pos_distances))


std_val = np.std(neg_distances)
avr_val = np.average(neg_distances)
print("Negative sub-patch cnt: %d " % len(neg_distances))
print("Negative Average: %.4f, STD: %.4f" % (avr_val, std_val))

std_val = np.std(pos_distances)
avr_val = np.average(pos_distances)
print("Positive sub-patch cnt: %d " % len(pos_distances))
print("Positive Average: %.4f, STD: %.4f" % (avr_val, std_val))

neg_sub_patches_cnt = len(neg_distances)
pos_sub_patches_cnt = len(pos_distances)

###########################################
# sort assignments by frequency
# neg_counter = collections.Counter(neg_distances)
neg_counter = {}
for k in neg_distances:
    neg_counter[k] = neg_counter.get(k, 0) + 1

# pos_counter = collections.Counter(pos_distances)
pos_counter = {}
for k in pos_distances:
    pos_counter[k] = pos_counter.get(k, 0) + 1

sorted_x = sorted(neg_counter.items(), key=operator.itemgetter(1))  # sort on keys:key=operator.itemgetter(1)
neg_freq = []
pos_freq = []
x_ticks = []
for s in sorted_x:
    neg_freq.append(s[1])
    pos_freq.append(pos_counter.get(s[0]))
    x_ticks.append(s[0])

pos_freq_smooth = savgol_filter(pos_freq, 51, 3)
neg_freq_plus = [x+200 for x in neg_freq]
pos_freq_smooth_plus = [x+200 for x in pos_freq_smooth]


x = np.arange(0, len(sorted_x), 1)
plt.figure()
plt.plot(x, neg_freq, x, pos_freq, x, pos_freq_smooth, x, neg_freq_plus, x, pos_freq_smooth_plus)
plt.xlim([0,len(sorted_x)])
plt.ylim([0, 700])
plt.legend(["Neg", "Pos", "Pos_smooth", "Neg+200", "Pos_smooth+200"])
plt.title('P%d, N%d' % (len(pos_distances), len(neg_distances)))
plt.xlabel('Assignments')
plt.ylabel('Frequency')
plt.grid()

plt.savefig(os.path.join(data_root_dir,"sorted_assignments.png"))
###########################################

#
# # bins_no = max(len(set(neg_distances)), len(set(pos_distances)))
# plt.figure(1)
# plt.hist([neg_distances, pos_distances], color=['orange', 'blue'], bins=9980)
# plt.legend(["Negative","Positive"], loc='upper center')
#
# plt.grid()
# # plt.hist([neg_distances, pos_distances], bins="auto", density=True, color=['orange', 'blue'], alpha=0.75, linewidth=2, histtype='step')
# # plt.legend(["Positive","Negative"], loc='upper center')# be aware! reversed legend.
# # Explaination: https://stackoverflow.com/questions/47084606/column-order-reversed-in-step-histogram-plot
# title_str = ("Dist distribution. N%d,P%d,Sub_N%d,Sub_P%d" % (neg_patch_cnt, pos_patch_cnt, neg_sub_patches_cnt, pos_sub_patches_cnt))
# plt.title(title_str)
# sn_str = ("DD_N%d-P%d-Sub_N%d-Sub_P%d.jpg" % (neg_patch_cnt, pos_patch_cnt, neg_sub_patches_cnt, pos_sub_patches_cnt))
# plt.savefig(os.path.join(data_root_dir, sn_str))



















