from matplotlib import pyplot
from libKMCUDA import kmeans_cuda
import numpy as np
import os

cluster_num = 10000

# representation_data_dir = "/projects/shart/digital_pathology/data/biliary/WSI_Neg_encode/CaseSubPatches_encode"
# kmeans_out_dir = "/projects/shart/digital_pathology/data/biliary/models/BOW/CaseSubPatches_encode_kmeans"

representation_data_dir = "/dtascfs/m192500/BiliaryCytology/CaseSubPatches_encode"
kmeans_out_dir = "/dtascfs/m192500/BiliaryCytology/CaseSubPatches_encode_kmeans"

def load_representation_code(representation_data_dir):
    representation_sub_patch_codes = np.empty([0, 144])
    # load an image from a WSI case
    representation_data_files = os.listdir(representation_data_dir)
    print("loading representation space")
    for r_file in representation_data_files:
        rp_codes = np.load(os.path.join(representation_data_dir, r_file))
        representation_sub_patch_codes = np.vstack((representation_sub_patch_codes, rp_codes))
    return representation_sub_patch_codes

samples = load_representation_code(representation_data_dir)

centroids, assignments, avg_dist = kmeans_cuda(samples.astype(np.float32), cluster_num, yinyang_t=0, average_distance=True, verbosity=1, seed=3)

# delete NaNs in centroids and assignments
mask = ~np.isnan(centroids).any(axis=1)
centroids = centroids[mask]
cmap = np.full(len(mask), -1, dtype=int)
for i, x in enumerate(np.where(mask)):
    cmap[x] = i
for i, ass in enumerate(assignments):
    assignments[i] = cmap[ass]

# use the centroids to initiate the KNN methods, do Kmeans again
cluster_num = len(centroids)
centroids, assignments, avg_dist = kmeans_cuda(samples.astype(np.float32), len(centroids), tolerance=1, average_distance=True, init=centroids, yinyang_t=0)

# delete NaNs in centroids and assignments
mask = ~np.isnan(centroids).any(axis=1)
centroids = centroids[mask]
cmap = np.full(len(mask), -1, dtype=int)
for i, x in enumerate(np.where(mask)):
    cmap[x] = i
for i, ass in enumerate(assignments):
    assignments[i] = cmap[ass]

cluster_num = len(centroids)

np.save(os.path.join(kmeans_out_dir, "centroids_samples_" + str(cluster_num) + ".npy"), centroids)
np.save(os.path.join(kmeans_out_dir, "assignments_samples_" + str(cluster_num) + ".npy"), assignments)
np.save(os.path.join(kmeans_out_dir, "avg_dist_samples_" + str(cluster_num) + ".npy"), avg_dist)


