#ifndef FCM_CUDA_CUH
#define FCM_CUDA_CUH


__host__ void init_membership(float ***i_membership);

__host__ void init_centers(float *i_cluster_centers);

__device__ float eucl_distance(float center, float val);

__device__ void update_centers_kernel(float **i_image, float ***i_membership, int i_rows, int i_cols, int i_num_clutsers);

__device__ void update_membership_kernel(float **i_image, float *i_cluster_centers, float ***i_membership, int i_rows, int i_cols, int i_num_clutsers);

__device__ float calculate_membership_point_kernel(float **i_image, float *i_cluster_centers, int i, int j, int k, int i_num_clutsers);

__host__ fcm_step(float **i_image, float ***i_membership, int rows, int cols, int T, int i_num_clutsers);

__device__ void calculate_final_cluster_kernel(float ***i_membership, int **i_final_cluster, int i_num_clutsers);