#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <random>
#include "fcm_cuda.cuh"

using namespace std;

__host__ void init_membership(float *i_membership, int i_rows, int i_cols, int i_num_clutsers) {
    // cout << i_cols << endl;
    // cout << i_rows << endl;

    // 1d-configuration
    int len = i_rows * i_cols * i_num_clutsers;
    i_membership = new float[len];
    for (int i = 0; i < len; ++i) {
        i_membership[i] = 1 / (float)i_num_clutsers;
    }

    // 3d-configuration
    // i_membership = new float** [i_rows];

    // for (int i = 0; i < i_rows; ++i) {
    //     i_membership[i] = new float* [i_cols];
    //     // i_new_membership[i] = new float* [i_cols];
    //     for (int j = 0; j < i_cols; ++j) {
    //         i_membership[i][j] = new float[i_num_clutsers];
    //         // i_new_membership[i][j] = new float[i_num_clutsers];
    //     }
        
    // }

    // for (int i = 0; i < i_rows; ++i) {
    //     for (int j = 0; j < i_cols; ++j) {
    //         for (int k = 0; k < i_num_clutsers; ++k) {
    //             i_membership[i][j][k] = 1 / (float)i_num_clutsers;
    //             // i_new_membership[i][j][k] = 99999;
    //         }
    //     }
    // }

}

__host__ void init_centers(float *i_cluster_centers, int i_num_clutsers) {
    i_cluster_centers = new float[i_num_clutsers];

    for (int i = 0; i < i_num_clutsers; ++i) {
        // randomly select i_num_clutsers points as cluster centers

        // random generator
        random_device rd;
        mt19937 eng(rd());
        uniform_real_distribution<> dist(0, 1);

        i_cluster_centers[i] = dist(eng);
    }

    // cout << "Centers: " << endl;
    // for (int i = 0; i < i_num_clutsers; ++i) {
    //     cout << i_cluster_centers[i] << endl;
    // }
}

__host__ void init_final_cluster(int* i_final_cluster, int rows, int cols) {
    i_final_cluster = new int[rows * cols];
    for (int i  = 0; i < rows * cols; ++i) {
        i_final_cluster[i] = -1;
    }
}

__device__ float eucl_distance(float center, float val) {
    // val: data point value
    // i: cluster center point value
    return sqrt(pow(val - center, 2));
}

__global__ void update_centers_kernel(float *i_image, float *i_membership, float *i_cluster_centers, int i_rows, int i_cols, int i_num_clutsers, int i_m) {
    float u_ij_m, x_u_ij_m;

    for (int k = 0; k < i_num_clutsers; ++k) {
        u_ij_m = 0.0, x_u_ij_m = 0.0;
        for (int i = 0; i < i_rows; ++i) {
            for (int j = 0; j < i_cols; ++j) {
                // FIX!!
                // u_ij_m += pow(i_membership[i][j][k], i_m);
                // x_u_ij_m += i_image[i][j] * pow(i_membership[i][j][k], i_m);
            }
        }
        // cout << "u, x: " << u_ij_m << " " << x_u_ij_m << endl;
        // cout << "c1: " << i_cluster_centers[k] << " ";
        i_cluster_centers[k] = x_u_ij_m / u_ij_m;
        // cout << "c2: " << i_cluster_centers[k] << endl;

    }

    // cout << "Centers: ";
    // for (int i = 0; i < i_num_clutsers; ++i) {
    //     cout << i_cluster_centers[i] << " ";
    // }
    // cout <<endl;

}


__global__ void update_membership_kernel(float *i_image, float *i_cluster_centers, float *i_membership, int i_rows, int i_cols, int i_num_clutsers, int i_m) {

    // calculate degree of membership of each data point (image) regarding each cluster
    for (int i = 0; i < i_rows; ++i) {
        for (int j = 0; j < i_cols; ++j) {
            for (int k = 0; k < i_num_clutsers; ++k) {
                // cout << "Hi" << endl;
                // i_new_membership[i][j][k] = calculate_membership_point(i, j, k);
                // FIX!!
                // i_membership[i][j][k] = calculate_membership_point_kernel(i_image, i_cluster_centers, i, j, k, i_num_clutsers, i_m);
            }
        }
    }

    // shared memory that stores: memberships for that pixel, 
    extern __shared__ int shared_d[];

    // print_mebership();

}


__device__ float calculate_membership_point_kernel(float *i_image, float *i_cluster_centers, int i, int j, int k, int i_num_clutsers, int i_m) {
    float d_center = 0, d_all = 0, aggr = 0.0;

    // FIX!!
    // d_center = eucl_distance(i_cluster_centers[k], i_image[i][j]);
    // cout << "d_center: " << d_center << endl;
    for (int c = 0; c < i_num_clutsers; ++c) {
        // FIX!!
        // d_all = eucl_distance(i_cluster_centers[c], i_image[i][j]);
        // cout << "d_all " << c << ": " << d_all << endl;
        // cout << "center " << c << ": " << i_cluster_centers[c] << endl;
        aggr += pow((d_center / d_all), 2 / (i_m - 1));
    }

    // cout << "Aggr: " << aggr << endl;

    return 1.0 /aggr;
}

__host__ void fcm_step(float *i_image, float *i_membership, float *i_cluster_centers, int rows, int cols, int T, int i_num_clutsers, int i_m, int threads_per_block) {
    // set block size and number of blocks to allocate
    dim3 dimBlock(threads_per_block, threads_per_block);
    dim3 dimGrid(((rows - 1) / dimBlock.x) + 1, ((cols - 1) / dimBlock.y) + 1);
    unsigned int shared_size_m = ((1 + 2 * i_num_clutsers) * threads_per_block * threads_per_block) * sizeof(float);
    unsigned int shared_size_c = (i_num_clutsers * threads_per_block * threads_per_block) * sizeof(float);
    // iteratively update membership and centers for T epochs
    for (int i = 0; i < T; ++i) {
        // fcm_step_kernel<<<dimGrid, dimBlock, shared_size>>>(i_image, i_membership, i_cluster_centers, rows, cols, i_num_clutsers, i_m);
        update_membership_kernel<<<dimGrid, dimBlock, shared_size_m>>>(i_image, i_cluster_centers, i_membership, rows, cols, i_num_clutsers, i_m);
        cudaDeviceSynchronize();
        update_centers_kernel<<<dimGrid, dimBlock, shared_size_c>>>(i_image, i_membership, i_cluster_centers, rows, cols, i_num_clutsers, i_m);
        cudaDeviceSynchronize();
    }
    // fcm_step_kernel<<<dimGrid, dimBlock, shared_size>>>(i_image, i_membership, i_cluster_centers, rows, cols, i_num_clutsers, i_m);
}

__global__ void fcm_step_kernel(float *i_image, float *i_membership, float *i_cluster_centers, int rows, int cols, int i_num_clutsers, int i_m) {
    // update_membership_kernel(i_image, i_cluster_centers, i_membership, rows, cols, i_num_clutsers, i_m);
    // update_centers_kernel(i_image, i_membership, i_cluster_centers, rows, cols, i_num_clutsers, i_m);
}


__device__ void calculate_final_cluster_kernel(float *i_membership, int **i_final_cluster, int i_num_clutsers, int i_rows, int i_cols) {
    // cout << "Membership: " << endl;
    for (int i = 0; i < i_rows; ++i) {
        for (int j = 0; j < i_cols; ++j) {
            float tmp_max = -999;
            for (int k = 0; k < i_num_clutsers; ++k) {
                // FIX!!
                // if (i_membership[i][j][k] >= tmp_max) {
                //     // cout << "hi" << endl;
                //     tmp_max = i_membership[i][j][k];
                //     // i_final_cluster[i][j] = 255 / (float)(k + 1);
                //     i_final_cluster[i][j] = k;
                // }
                int k = 0;
                // cout << i_membership[i][j][k] << " ";
            }
            // cout << endl;
        }
    }
}

