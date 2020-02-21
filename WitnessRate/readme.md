## Background:

Witness rate (WR) is the ratio of the number of positive instances in a positive bag to the size of the
bag. Many MIL problems in the real world naturally have positive bags with low witness rate (Zhou
et al., 2009; Zhang and Goldman, 2002). So it is possible to use the witness rate to measure the 
confidence of a bag for identifying positive events.

## Descriptions of this code:
This method is based on sub-patch voting strategy. The entire workflow consists of the following steps:
1. Preprocess (files are in ./patch_extraction/*)  
    a) extract image patches and filter informativeness ones (script in: extract_all.bat)
      the script is created by WSI_extraction_sh_creation.py, and it will call WSI_extraction_sh.py which depends on WSI_extraction_utils.py
    b) image normalization (WSI_patches_norm.py)  
2. Image representation (files are in ./subpatch_representation/*)  
    a) split image into sub-patches (WSI_Neg_numpy_dense_mp.py)
    b) use [Google's image compression](https://github.com/tensorflow/models/tree/master/research/compression) to encode sub-patches, use the code as the representation vector (WSI_Neg_sub_patch_encoder.py)
3. Patch Scoring model (files are in ./patch_scoring/*) 
    a) KMeans (Did on DSVM with GPU accelerated KNN: [CUDAKmeans](https://github.com/src-d/kmcuda), saved as centroids_samples_#K.npy )
        step 1). clustering 80% negative samples into 10000 clusters; (sub_patches_kmeans.py)   
    b) Clusters weighting (draw intermediate result) (train.py)
        step 2). assign labels to the rest 20% negative samples and all the 'positive' samples with the KNN model trained in the previous step; 
        step 3). calculate proportion of negative samples in each cluster 
    c) definition of the model. 
    Information should be saved in model:centroids_samples_#_.npy, weights for each cluster
4. Evaluation (files are in ./validation/*)   
    a) examples:   
        i) validate on overall dataset (BOW_weighting.py)   
        ii) validate on annotated (eval.py)   
    b) general steps:   
        i) Get representation   
        ii) Patch scoring by sub-patches' voting
                
### References:
[1] Carbonneau, Marc-André, Veronika Cheplygina, Eric Granger, and Ghyslain Gagnon. "Multiple instance learning: A survey of problem characteristics and applications." Pattern Recognition 77 (2018): 329-353.   
[2] Q. Zhang and S. Goldman. EM-DD: An improved multiple-instance learning technique. In Advances in Neural Information Processing Systems 14, pages 1073–1080, Cambridge, MA, 2002. MIT Press.   
[3] Z.-H. Zhou, Y.-Y. Sun, and Y.-F. Li. Multi-instance learning by treating instances as non-i.i.d. samples. In the 26th International Conference on Machine Learning, pages 1249–1256, 2009.   
[4] Liu, Guoqing, Jianxin Wu, and Zhi-Hua Zhou. "Key instance detection in multi-instance learning." (2012).
       