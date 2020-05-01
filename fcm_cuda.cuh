__host__ void init_membership(float *i_membership, int i_rows, int i_cols, int i_num_clutsers);

__host__ void init_centers(float *i_cluster_centers, int i_num_clutsers);

__device__ float eucl_distance(float center, float val);

__global__ void update_centers_kernel(float *i_image, float *i_membership, float *i_cluster_centers, int i_rows, int i_cols, int i_num_clutsers, int i_m);

__global__ void update_membership_kernel(float *i_image, float *i_cluster_centers, float *i_membership, int i_rows, int i_cols, int i_num_clutsers, int i_m);

__device__ float calculate_membership_point_kernel(float *i_image, float *i_cluster_centers, int i, int j, int k, int i_num_clutsers, int i_m);

__host__ void fcm_step(float *i_image, float *i_membership, float *i_cluster_centers, int rows, int cols, int T, int i_num_clutsers, int i_m, int threads_per_block);

__global__ void fcm_step_kernel(float *i_image, float *i_membership, float *i_cluster_centers, int rows, int cols, int i_num_clutsers, int i_m);

__host__ void init_final_cluster(int* i_final_cluster, int rows, int cols);

__device__ void calculate_final_cluster_kernel(float *i_membership, int **i_final_cluster, int i_num_clutsers, int i_rows, int i_cols);