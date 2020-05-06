#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <random>

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "fcm_thrust.cuh"

using namespace std;

__host__ void init_membership(float *i_membership, int i_rows, int i_cols, int i_num_clutsers) {
    // cout << i_cols << endl;
    // cout << i_rows << endl;
    // cout << "Init m start"  << endl;

    // 1d-configuration
    int len = i_rows * i_cols * i_num_clutsers;
    i_membership = new float[len];
    for (int i = 0; i < len; ++i) {
        i_membership[i] = 1 / (float)i_num_clutsers;
    }

    // cout << "Init m done"  << endl;

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
    // cout << "Init c start"  << endl;
    i_cluster_centers = new float[i_num_clutsers];

    for (int i = 0; i < i_num_clutsers; ++i) {
        // randomly select i_num_clutsers points as cluster centers

        // random generator
        random_device rd;
        mt19937 eng(rd());
        uniform_real_distribution<> dist(0, 1);

        i_cluster_centers[i] = dist(eng);
    }

    // cout << "Init c done"  << endl;

    // cout << "Centers: " << endl;
    // for (int i = 0; i < i_num_clutsers; ++i) {
    //     cout << i_cluster_centers[i] << endl;
    // }
}

__host__ void init_final_cluster(int* i_final_cluster, int rows, int cols) {
    // cout << "Init f start"  << endl;
    // i_final_cluster = new int[rows * cols];
    for (int i  = 0; i < rows * cols; ++i) {
        i_final_cluster[i] = -1;
    }
    // cout << "Init f done"  << endl;
}

__device__ float eucl_distance(float center, float val) {
    // val: data point value
    // i: cluster center point value
    return sqrt(pow(val - center, 2));
}

__global__ void update_centers_numerator_kernel(float *i_image, float *i_membership, float *i_cluster_centers, int i_rows, int i_cols, int i_num_clutsers, int i_m, float* numerator, int c) {
    // extern __shared__ float shared_d[];
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = global_idx % i_rows;
    int j = (global_idx - i) / i_rows;
    int k = c;

    numerator[j * i_rows + i] = i_image[i + i_rows * j] * pow(i_membership[k * i_rows * i_cols + j * i_rows + i], i_m); 
}

__global__ void update_centers_denominator_kernel(float *i_image, float *i_membership, float *i_cluster_centers, int i_rows, int i_cols, int i_num_clutsers, int i_m, float* denominator, int c) {
    // printf("222\n");
    // extern __shared__ float shared_d[];
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = global_idx % i_rows;
    int j = (global_idx - i) / i_rows;
    int k = c;

    denominator[j * i_rows + i] = pow(i_membership[k * i_rows * i_cols + j * i_rows + i], i_m);
}


__global__ void update_membership_kernel(float *i_image, float *i_cluster_centers, float *i_membership, int i_rows, int i_cols, int i_num_clutsers, int i_m) {
    // std::printf("Dafuq?\n");

    // shared memory that stores: memberships for that pixel, 
    extern __shared__ float shared_d[];
    // std::printf("%f???\n", i_image[0]);

    // global index
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int per_thread_len = 2 + i_num_clutsers;

    // x, y, z indices in original membership matrix
    int x = global_idx % i_rows ;
    int y = ((global_idx - x) / i_rows) % i_cols;
    int z = ((global_idx - x - y * i_rows) / (i_rows * i_cols));


    // load data into shared memory (each thread loads info for a **single** pixel membership)
    if (global_idx < i_rows * i_cols * i_num_clutsers) {
        // load pixel (1)
        // std::printf("fxxk this sxit %f\n", i_image[global_idx]);
        shared_d[threadIdx.x * per_thread_len] = i_image[x + y * i_rows];
        // std::printf("fxxk this sxit %f\n", i_image[global_idx]);
        // load membership (1)
        shared_d[threadIdx.x * per_thread_len + 1] = i_membership[(z * i_rows * i_cols) + y * i_rows + x];
        // load centers (i_num_clutsers)
        for (int i = 0; i < i_num_clutsers; ++i) {
            // shared_d[threadIdx.x * per_thread_len + (i + 1)] = i_membership[(x + y * i_rows) * i_num_clutsers + i];
            shared_d[threadIdx.x * per_thread_len + 1 + (i + 1)] = i_cluster_centers[i];
        }
    }
    else {
        // std::printf("else???? %f\n", i_image[global_idx]);
        for (int i = 0; i < per_thread_len; ++i) {
            shared_d[threadIdx.x] = 0;
        }
    }

    
    // printf("image membership: %f %f \n", shared_d[threadIdx.x * per_thread_len], shared_d[threadIdx.x * per_thread_len + 1]);
    // printf("   centers: ");
    // for (int j = 0; j < i_num_clutsers; ++j) {
    //     printf("center %f", shared_d[threadIdx.x * per_thread_len + 1 + (j + 1)]);
    // }
    // printf("\n");
    

    __syncthreads();

    // std::printf("Dafuq?\n");

    // calculate membership for the loaded pixel
    float d_center = 0, d_all = 0, aggr = 0.0;
    d_center = eucl_distance(shared_d[threadIdx.x * per_thread_len + 1 + (z + 1)], shared_d[threadIdx.x * per_thread_len]);
    // std::printf("a, b: %f, %f\n", shared_d[threadIdx.x * per_thread_len + 1 + (z + 1)], shared_d[threadIdx.x * per_thread_len]);
    for (int c = 0; c < i_num_clutsers; ++c) {
        d_all = eucl_distance(shared_d[threadIdx.x * per_thread_len + 1 + (c + 1)], shared_d[threadIdx.x * per_thread_len]);
        aggr += pow((d_center / d_all), 2 / (i_m - 1));
        // printf("z c d_center d_all: %d %d %f %f \n", z, c, d_center, d_all);
    }

    // write aggregation results to membership value on shared memory
    // std::printf("Aggr: %f\n", aggr);
    // printf("z d_center d_all Aggr: %d %f %f %f\n", z, d_center, d_all, aggr);
    shared_d[threadIdx.x * per_thread_len + 1] = 1.0 / (float)aggr;
    // std::printf("PLZ %f\n", shared_d[threadIdx.x * per_thread_len + 1]);

    // write back membership results to global memory
    i_membership[(z * i_rows * i_cols) + y * i_rows + x] = 1.0 / (float)aggr;
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

__host__ void fcm_step(float *i_image, float *i_membership, float *i_cluster_centers, int rows, int cols, int T, int i_num_clutsers, int i_m, int threads_per_block, float *out_membership) {
    // set block size and number of blocks to allocate
    // dim3 dimBlock(threads_per_block, threads_per_block);
    // dim3 dimGrid(((rows - 1) / dimBlock.x) + 1, ((cols - 1) / dimBlock.y) + 1);

    unsigned int blks_m = 1 + (rows * cols * i_num_clutsers - 1) / threads_per_block;
    // unsigned int blks_c = 1 + (i_num_clutsers - 1) / threads_per_block;
    unsigned int shared_size_m = ((1 + 2 * i_num_clutsers) * threads_per_block) * sizeof(float);
    // unsigned int shared_size_c = (i_num_clutsers * threads_per_block) * sizeof(float);

    // cout << "blkm: " << blks_m << endl;
    // cout << "blkc: " << blks_c << endl;

    // // copy output memebership matrix
    // for (int i = 0; i < rows; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //         for (int k = 0; k < i_num_clutsers; ++k) {
    //             std::printf("%f\n", i_membership[(k * rows * cols) + j * rows + i]);
    //         }
    //     }
    // }

    // cudaMemcpy everything!
    float *d_image, *d_membership, *d_cluster_centers;
    cudaMalloc((void**)&d_image, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_membership, rows * cols * i_num_clutsers * sizeof(float));
    cudaMalloc((void**)&d_cluster_centers, i_num_clutsers * sizeof(float));
    // cudaMalloc((void**)&out_membership, rows * cols * i_num_clutsers * sizeof(float));

    cudaMemcpy(d_image, i_image, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_membership, i_membership, rows * cols * i_num_clutsers * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_centers, i_cluster_centers, i_num_clutsers * sizeof(float), cudaMemcpyHostToDevice);

    // kernel config parameters for reduction
    unsigned int blks_c = 1 + (rows * cols - 1) / threads_per_block;
    unsigned int in_size = rows * cols;
    // unsigned int shared_size_c = threads_per_block * sizeof(float);

    // memory allocations for reduction
    // float *numerator = new float[1];
    // float *denominator = new float[1];
    float numerator, denominator;
    // float *device_in_num, *device_in_den, *device_out_num, *device_out_den;

    thrust::device_vector<float> device_in_num(in_size);
    thrust::device_vector<float> device_in_den(in_size);

    // obtain raw pointer to device vector’s memory
    float * num_ptr = thrust::raw_pointer_cast(&device_in_num[0]);
    float * den_ptr = thrust::raw_pointer_cast(&device_in_den[0]);

    // cudaMalloc((void **)&device_in_num, in_size * sizeof(float));
    // cudaMalloc((void **)&device_in_den, in_size * sizeof(float));
    // cudaMalloc((void **)&device_out_num, blks_c * sizeof(float));
    // cudaMalloc((void **)&device_out_den, blks_c * sizeof(float));


    // iteratively update membership and centers for T epochs
    for (int i = 0; i < T; ++i) {
        // cout << "Hi" << endl;
        // fcm_step_kernel<<<dimGrid, dimBlock, shared_size>>>(i_image, i_membership, i_cluster_centers, rows, cols, i_num_clutsers, i_m);
        // 1D grid of 2D blocks
        // printf("1\n");
        update_membership_kernel<<<blks_m, threads_per_block, shared_size_m>>>(d_image, d_cluster_centers, d_membership, rows, cols, i_num_clutsers, i_m);
        // printf("2\n");
        // cudaDeviceSynchronize();
        // reduce each center with numerator and denominator reduction
        
        // cudaMemcpy(device_in_num, i_membership, in_size * sizeof(float), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(device_in_den, i_membership, in_size * sizeof(float), cudaMemcpyDeviceToDevice);
        for (int j = 0; j < i_num_clutsers; ++j) {
            blks_c = 1 + (rows * cols - 1) / threads_per_block;
            // do thrust oprations here
            // printf("3\n");
            update_centers_numerator_kernel<<<blks_c, threads_per_block>>>(d_image, d_membership, d_cluster_centers, rows, cols, i_num_clutsers, i_m, num_ptr, j);
            // printf("4\n");
            update_centers_denominator_kernel<<<blks_c, threads_per_block>>>(d_image, d_membership, d_cluster_centers, rows, cols, i_num_clutsers, i_m, den_ptr, j);
            // printf("5\n");
            numerator = thrust::reduce(device_in_num.begin(), device_in_num.end());
            denominator = thrust::reduce(device_in_den.begin(), device_in_den.end());
            // printf("6\n");
            // printf("nn: %f\n", numerator);
            // printf("dd: %f\n", denominator);
            // cudaDeviceSynchronize();

            i_cluster_centers[j] = numerator / denominator;
        }
        // cudaMemcpy(d_cluster_centers, i_cluster_centers, i_num_clutsers * sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_membership, i_membership, rows * cols * i_num_clutsers * sizeof(float), cudaMemcpyHostToDevice);

        // update_centers_kernel<<<blks_c, threads_per_block, shared_size_c>>>(d_image, d_membership, d_cluster_centers, rows, cols, i_num_clutsers, i_m, threads_per_block);
        // cudaDeviceSynchronize();
        // cout << "Bye" << endl;
    }
    // fcm_step_kernel<<<dimGrid, dimBlock, shared_size>>>(i_image, i_membership, i_cluster_centers, rows, cols, i_num_clutsers, i_m);

    cudaMemcpy(out_membership, d_membership, rows * cols * i_num_clutsers * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(i_cluster_centers, d_cluster_centers, i_num_clutsers * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_membership);
    cudaFree(d_cluster_centers);
}

__global__ void fcm_step_kernel(float *i_image, float *i_membership, float *i_cluster_centers, int rows, int cols, int i_num_clutsers, int i_m) {
    // update_membership_kernel(i_image, i_cluster_centers, i_membership, rows, cols, i_num_clutsers, i_m);
    // update_centers_kernel(i_image, i_membership, i_cluster_centers, rows, cols, i_num_clutsers, i_m);
}


__host__ void calculate_final_cluster(float *i_membership, int *i_final_cluster, int i_num_clutsers, int i_rows, int i_cols) {
    for (int i = 0; i < i_rows; ++i) {
        for (int j = 0; j < i_cols; ++j) {
            float tmp_max = -9999;
            // std::printf("[");
            for (int k = 0; k < i_num_clutsers; ++k) {
                // std::printf("%f ", i_membership[(k * i_rows * i_cols) + j * i_rows + i]);
                if (i_membership[(k * i_rows * i_cols) + j * i_rows + i] >= tmp_max) {
                    tmp_max = i_membership[(k * i_rows * i_cols) + j * i_rows + i];
                    i_final_cluster[j * i_rows + i] = k;
                }
            }
            // std::printf("]");
        }
        // std::printf("\n");
    }

}

