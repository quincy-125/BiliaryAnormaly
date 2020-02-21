from config.load_config import LoadConfig
import os, sys
import numpy as np
from sklearn.metrics import pairwise_distances
import pickle
import matplotlib.pyplot as plt


def get_representations(patch_fn, code_folder):
    fn = os.path.split(patch_fn)[1][:-4]
    encode_fn = os.path.join(code_folder, fn+"_sub_encode.npy")
    if os.path.exists(encode_fn):
        sub_patch_codes = np.load(encode_fn)
    else:
        # TODO: use Google's image representation to represent image
        #raise Exception("unable to load the patch representation")
        return None
    return sub_patch_codes


def get_closest_cluster(compare_nodes, reference_nodes):
    dist_matrix = pairwise_distances(compare_nodes.astype(np.float32), reference_nodes.astype(np.float32))
    min_index = np.argmin(dist_matrix, axis=1)
    min_dists = np.amin(dist_matrix, axis=1)
    return min_index, min_dists


def sub_patch_voting(sub_patch_codes, centroids, cluster_weights):
    assignments, distances = get_closest_cluster(sub_patch_codes, centroids)
    patch_score = []
    for sub_ass in assignments:
        patch_score.append(cluster_weights.get(sub_ass, 0))
    top_scores1 = max(patch_score)
    patch_score.remove(top_scores1)
    top_scores2 = max(patch_score)
    return top_scores2 + top_scores1
    # return top_scores1



def load_cluster_weights(fn):
    with open(fn, 'rb') as handle:
        return pickle.load(handle)


def eval_img_folder(img_dir, testing_img_rep_dir,eval_save_to, knn_centroids_fn, train_log_dir):
    imgs = os.listdir(img_dir)
    cluster_weights = load_cluster_weights(os.path.join(train_log_dir, 'cluster_weights.pickle'))
    centroids = np.load(os.path.join(knn_centroids_fn))
    fp = open(eval_save_to, 'a')
    patch_scores = []
    for img in imgs:
        img_fn = os.path.join(img_dir, img)
        sub_patch_codes = get_representations(img_fn, testing_img_rep_dir)
        if sub_patch_codes is not None:
            patch_score = sub_patch_voting(sub_patch_codes, centroids, cluster_weights)
            patch_scores.append(patch_score)
            fp.write(img + "," + str(patch_score) + "\n")
            # print(patch_score)
        else:
            print("Unable to find the representation")
    fp.close()
    return patch_scores


if __name__ == '__main__':
    cwd = sys.path
    conf = LoadConfig(os.path.join(cwd[0], 'config//model_cfg.yml'))
    benign_eval_res = os.path.join(conf.eval_inst_sv, 'benign.csv')
    benign_patch_score = eval_img_folder(conf.annotation_benign_dir, conf.testing_img_rep_dir, benign_eval_res, conf.knn_centroids_fn, conf.train_log_dir)
    malignant_eval_res = os.path.join(conf.eval_inst_sv, 'malignant.csv')
    malignant_patch_score = eval_img_folder(conf.annotation_malignant_dir, conf.testing_img_rep_dir, malignant_eval_res, conf.knn_centroids_fn, conf.train_log_dir)
    uninformative_eval_res = os.path.join(conf.eval_inst_sv, 'uninformative.csv')
    uninformative_patch_score = eval_img_folder(conf.annotation_uninformative_dir, conf.testing_img_rep_dir, uninformative_eval_res,  conf.knn_centroids_fn, conf.train_log_dir)
    gray_zone_eval_res = os.path.join(conf.eval_inst_sv, 'gray_zone.csv')
    gray_zone_patch_score = eval_img_folder(conf.annotation_grayzone_dir, conf.testing_img_rep_dir, gray_zone_eval_res, conf.knn_centroids_fn, conf.train_log_dir)

    bins_no = 500
    plt.figure(1)
    # plt.hist([benign_patch_score, malignant_patch_score, uninformative_patch_score, gray_zone_patch_score], bins=bins_no, density=True, color=['blue','orange', 'red', 'green'],
             # alpha=0.75, linewidth=2, histtype='step')
    # plt.hist([benign_patch_score, malignant_patch_score, uninformative_patch_score, gray_zone_patch_score],
    #          bins=bins_no, density=True, alpha=0.75, linewidth=2, histtype='step')
    # plt.grid()
    # plt.legend(["Benign", "Malignant", "Uninformative", "gray_zone_patch_score"])
    # plt.title("Score distribution")
    # plt.savefig(os.path.join(conf.eval_inst_sv, "Score_distribution.jpg"))

    plt.hist([benign_patch_score, malignant_patch_score],
             bins=bins_no, density=True, alpha=0.75, linewidth=2, histtype='step')
    plt.grid()
    plt.legend(["Benign", "Malignant"])
    plt.title("Score distribution")
    plt.savefig(os.path.join(conf.eval_inst_sv, "Score_distribution_max_2.jpg"))




