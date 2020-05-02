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
    // float u_ij_m, x_u_ij_m;

    // for (int k = 0; k < i_num_clutsers; ++k) {
    //     u_ij_m = 0.0, x_u_ij_m = 0.0;
    //     for (int i = 0; i < i_rows; ++i) {
    //         for (int j = 0; j < i_cols; ++j) {
    //             // FIX!!
    //             // u_ij_m += pow(i_membership[i][j][k], i_m);
    //             // x_u_ij_m += i_image[i][j] * pow(i_membership[i][j][k], i_m);
    //         }
    //     }
    //     // cout << "u, x: " << u_ij_m << " " << x_u_ij_m << endl;
    //     // cout << "c1: " << i_cluster_centers[k] << " ";
    //     i_cluster_centers[k] = x_u_ij_m / u_ij_m;
    //     // cout << "c2: " << i_cluster_centers[k] << endl;

    // }

    // cout << "Centers: ";
    // for (int i = 0; i < i_num_clutsers; ++i) {
    //     cout << i_cluster_centers[i] << " ";
    // }
    // cout <<endl;

    // shared memory that stores: the centers
    extern __shared__ int shared_d[];
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float u_ij_m = 0, x_u_ij_m = 0;

    int x = global_idx % i_rows ;
    int y = ((global_idx - x) / i_rows) % i_cols;
    int z = ((global_idx - x - y) / (i_rows * i_cols));


    for (int i = 0; i < i_rows; ++i) {
        for (int j = 0; j < i_cols; ++j) {
            u_ij_m += pow(i_membership[z * i_rows * i_cols + j * i_rows + i], i_m);
            x_u_ij_m += i_image[i + i_rows * j] * pow(i_membership[z * i_rows * i_cols + j * i_rows + i], i_m);
        }
    }

    // store new center value in shared memory
    shared_d[global_idx] = x_u_ij_m / u_ij_m;

    // store center back to global memory
    i_cluster_centers[global_idx] = shared_d[global_idx];
}


__global__ void update_membership_kernel(float *i_image, float *i_cluster_centers, float *i_membership, int i_rows, int i_cols, int i_num_clutsers, int i_m) {

    // calculate degree of membership of each data point (image) regarding each cluster
    // for (int i = 0; i < i_rows; ++i) {
    //     for (int j = 0; j < i_cols; ++j) {
    //         for (int k = 0; k < i_num_clutsers; ++k) {
    //             // cout << "Hi" << endl;
    //             // i_new_membership[i][j][k] = calculate_membership_point(i, j, k);
    //             // FIX!!
    //             // i_membership[i][j][k] = calculate_membership_point_kernel(i_image, i_cluster_centers, i, j, k, i_num_clutsers, i_m);
    //         }
    //     }
    // }

    // shared memory that stores: memberships for that pixel, 
    extern __shared__ int shared_d[];

    // global index
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // chunk of shared memory each thread gets to use
    int per_thread_len = 2 + i_num_clutsers;
    // x, y, z indices in original membership matrix
    int x = global_idx % i_rows ;
    int y = ((global_idx - x) / i_rows) % i_cols;
    int z = ((global_idx - x - y) / (i_rows * i_cols));

    // load data into shared memory (each thread loads info for a **single** pixel)
    if (global_idx < i_rows * i_cols * i_num_clutsers) {
        // load pixel (1)
        shared_d[threadIdx.x * per_thread_len] = i_image[global_idx];
        // load membership (1)
        shared_d[threadIdx.x * per_thread_len + 1] = i_membership[(x + y * i_rows) * i_num_clutsers + z];
        // load centers (i_num_clutsers)
        for (int i = 0; i < i_num_clutsers; ++i) {
            // shared_d[threadIdx.x * per_thread_len + (i + 1)] = i_membership[(x + y * i_rows) * i_num_clutsers + i];
            shared_d[threadIdx.x * per_thread_len + 1 + (i + 1)] = i_cluster_centers[i];
        }
    }
    // pad remaining threads
    else {
        for (int i = 0; i < per_thread_len; ++i) {
            shared_d[threadIdx.x] = 0;
        }
    }
    __syncthreads();

    // calculate membership for the loaded pixel
    float d_center = 0, d_all = 0, aggr = 0.0;
    d_center = eucl_distance(shared_d[threadIdx.x * per_thread_len + 1 + (z + 1)], shared_d[threadIdx.x * per_thread_len]);
    for (int c = 0; c < i_num_clutsers; ++c) {
        d_all = eucl_distance(shared_d[threadIdx.x * per_thread_len + 1 + (c + 1)], shared_d[threadIdx.x * per_thread_len]);
        aggr += pow((d_center / d_all), 2 / (i_m - 1));
    }

    // write aggregation results to membership value on shared memory
    shared_d[threadIdx.x * per_thread_len + 1] = 1.0 / aggr;

    // write back membership results to global memory
    i_membership[(x + y * i_rows) * i_num_clutsers + z] = shared_d[threadIdx.x * per_thread_len + 1];
    __syncthreads();


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
    // dim3 dimBlock(threads_per_block, threads_per_block);
    // dim3 dimGrid(((rows - 1) / dimBlock.x) + 1, ((cols - 1) / dimBlock.y) + 1);
    unsigned int blks_m = 1 + (rows * cols * i_num_clutsers - 1) / threads_per_block;
    unsigned int blks_c = 1 + (i_num_clutsers - 1) / threads_per_block;
    unsigned int shared_size_m = ((1 + 2 * i_num_clutsers) * threads_per_block) * sizeof(float);
    unsigned int shared_size_c = (i_num_clutsers * threads_per_block) * sizeof(float);
    // iteratively update membership and centers for T epochs
    for (int i = 0; i < T; ++i) {
        // fcm_step_kernel<<<dimGrid, dimBlock, shared_size>>>(i_image, i_membership, i_cluster_centers, rows, cols, i_num_clutsers, i_m);
        // 1D grid of 2D blocks
        update_membership_kernel<<<blks_m, threads_per_block, shared_size_m>>>(i_image, i_cluster_centers, i_membership, rows, cols, i_num_clutsers, i_m);
        cudaDeviceSynchronize();
        update_centers_kernel<<<blks_c, threads_per_block, shared_size_c>>>(i_image, i_membership, i_cluster_centers, rows, cols, i_num_clutsers, i_m);
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

