from config.load_config import LoadConfig
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.signal import savgol_filter
import operator
import pickle

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


# Depend on three steps:
# a) sub-patch representations of all samples involved.
# b) KMeans on the 80% negative sub-patches
# c) Assignments on the rest 20% negative sub-patches
def clusters_weighting(neg_assignments_dir):
    neg_npy_files = os.listdir(neg_assignments_dir)
    neg_assignments = []
    for nf in neg_npy_files:
        neg_ass = np.load(os.path.join(neg_assignments_dir, nf))
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
    return cluster_weights


def draw_cluster_proportion(neg_mis_npy_dir, pos_mis_npy_dir, fig_save_name):
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
    neg_freq_plus = [x + 200 for x in neg_freq]
    pos_freq_smooth_plus = [x + 200 for x in pos_freq_smooth]

    x = np.arange(0, len(sorted_x), 1)
    plt.figure()
    plt.plot(x, neg_freq, x, pos_freq, x, pos_freq_smooth, x, neg_freq_plus, x, pos_freq_smooth_plus)
    plt.xlim([0, len(sorted_x)])
    plt.ylim([0, 700])
    plt.legend(["Neg", "Pos", "Pos_smooth", "Neg+200", "Pos_smooth+200"])
    plt.title('P%d, N%d' % (len(pos_distances), len(neg_distances)))
    plt.xlabel('Assignments')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(fig_save_name)


if __name__ == '__main__':
    cwd = sys.path
    conf = LoadConfig(os.path.join(cwd[0], 'config//model_cfg.yml'))

    save_cluster_weights = os.path.join(conf.train_log_dir, 'cluster_weights.pickle')
    cluster_weights = clusters_weighting(conf.neg_assignments_dir)
    with open(save_cluster_weights, 'wb') as handle:  # save the weights of the clusters into a model
        pickle.dump(cluster_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cluster_proportion_fig_save_name = os.path.join(conf.train_log_dir, "sorted_assignments.png")
    draw_cluster_proportion(conf.neg_assignments_dir, conf.pos_assignments_dir, cluster_proportion_fig_save_name)

    print(cluster_weights)
















