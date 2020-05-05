#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <random>
#include "fcm_stream.cuh"

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
    extern __shared__ float shared_d[];
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = global_idx % i_rows;
    int j = (global_idx - i) / i_rows;
    int k = c;

    // printf("1\n");

    // each thread loads one element into shared memory
    if (global_idx < i_rows * i_cols) {
        shared_d[threadIdx.x] = i_image[i + i_rows * j] * pow(i_membership[k * i_rows * i_cols + j * i_rows + i], i_m); 
    }
    else {
        shared_d[threadIdx.x] = 0;
    }
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_d[threadIdx.x] += shared_d[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // write result back to global memory
    if (threadIdx.x == 0) {
        numerator[blockIdx.x] = shared_d[0];
    }

}

__global__ void update_centers_denominator_kernel(float *i_image, float *i_membership, float *i_cluster_centers, int i_rows, int i_cols, int i_num_clutsers, int i_m, float* denominator, int c) {
    // printf("222\n");
    extern __shared__ float shared_d[];
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = global_idx % i_rows;
    int j = (global_idx - i) / i_rows;
    int k = c;

    // printf("2\n");

    // each thread loads one element into shared memory
    if (global_idx < i_rows * i_cols) {
        shared_d[threadIdx.x] = pow(i_membership[k * i_rows * i_cols + j * i_rows + i], i_m);
    }
    else {
        shared_d[threadIdx.x] = 0;
    }
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_d[threadIdx.x] += shared_d[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // write result back to global memory
    if (threadIdx.x == 0) {
        denominator[blockIdx.x] = shared_d[0];
    }
}

// __global__ void update_centers_kernel(float *i_image, float *i_membership, float *i_cluster_centers, int i_rows, int i_cols, int i_num_clutsers, int i_m, int threads_per_block) {
//     int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int blks = 1 + (i_rows * i_cols - 1) / threads_per_block;
//     unsigned int in_size = i_rows * i_cols;
//     unsigned int shared_size = threads_per_block * sizeof(float);
//     float *res_num = new float[blks];
//     float *res_den = new float[blks];
//     float *device_in_num, *device_in_den, *device_out_num, *device_out_den;
//     float numerator, denominator;

//     // cudaMalloc((void **)&device_in_num, in_size * sizeof(float));
//     // cudaMalloc((void **)&device_in_den, in_size * sizeof(float));
//     // cudaMalloc((void **)&device_out_num, blks * sizeof(float));
//     // cudaMalloc((void **)&device_out_den, blks * sizeof(float));
//     // cudaMemcpyAsync(device_in_num, i_membership, in_size * sizeof(float), cudaMemcpyDeviceToDevice);
//     // cudaMemcpyAsync(device_in_den, i_membership, in_size * sizeof(float), cudaMemcpyDeviceToDevice);

//     // while (true) {
//     //     update_centers_numerator_kernel<<<blks, threads_per_block, shared_size>>>(i_image, device_in_num, i_cluster_centers, i_rows, i_cols, i_num_clutsers, i_m, device_out_num, global_idx);
//     //     printf("g\n");
//     //     cudaDeviceSynchronize();
//     //     update_centers_denominator_kernel<<<blks, threads_per_block, shared_size>>>(i_image, device_in_den, i_cluster_centers, i_rows, i_cols, i_num_clutsers, i_m, device_out_den, global_idx);
//     //     cudaDeviceSynchronize();
//     //     printf("gg\n");
//     //     if (blks == 1) {
//     //         break;
//     //     }

//     //     in_size = blks;
//     //     // device_out now holds the reduce result, copy to device in and reduce again (next time)
//     //     cudaMemcpyAsync(device_in_num, device_out_num, in_size * sizeof(float), cudaMemcpyDeviceToDevice);
//     //     cudaMemcpyAsync(device_in_den, device_out_den, in_size * sizeof(float), cudaMemcpyDeviceToDevice);

//     //     blks = 1 + ((blks - 1) / threads_per_block);
//     // }

//     // cudaMemcpyAsync(res_num, device_out_num, sizeof(float), cudaMemcpyDeviceToDevice);
//     // cudaMemcpyAsync(res_den, device_out_den, sizeof(float), cudaMemcpyDeviceToDevice);

//     // numerator = res_num[0];
//     // denominator = res_den[0];

//     // update the cluster center (finally!)
//     // i_cluster_centers[global_idx] = numerator / denominator;
//     __syncthreads();

//     delete [] res_num;
//     delete [] res_den;
//     cudaFree(device_in_num);
//     cudaFree(device_in_den);
//     cudaFree(device_out_num);
//     cudaFree(device_out_den);
// }

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

    // std::printf("Dafuq?\n");

    // shared memory that stores: memberships for that pixel, 
    extern __shared__ float shared_d[];
    // std::printf("%f???\n", i_image[0]);

    // global index
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // std::printf("2?\n");
    // chunk of shared memory each thread gets to use
    int per_thread_len = 2 + i_num_clutsers;
    // x, y, z indices in original membership matrix
    int x = global_idx % i_rows ;
    int y = ((global_idx - x) / i_rows) % i_cols;
    int z = ((global_idx - x - y * i_rows) / (i_rows * i_cols));
    // std::printf("3?\n");


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

    // printf("d_center: %f\n", d_center);
    // printf("d_all: %f\n", d_all);

    // std::printf("Dafuq2?\n");

    // write aggregation results to membership value on shared memory
    // std::printf("Aggr: %f\n", aggr);
    // printf("z d_center d_all Aggr: %d %f %f %f\n", z, d_center, d_all, aggr);
    shared_d[threadIdx.x * per_thread_len + 1] = 1.0 / (float)aggr;
    // std::printf("PLZ %f\n", shared_d[threadIdx.x * per_thread_len + 1]);

    // write back membership results to global memory
    // std::printf("PLZ %f\n", shared_d[threadIdx.x * per_thread_len + 1]);
    i_membership[(z * i_rows * i_cols) + y * i_rows + x] = 1.0 / (float)aggr;
    // printf("Member: %f\n", i_membership[(z * i_rows * i_cols) + y * i_rows + x]);
    // std::printf("WHY %d %d %d %f\n", x, y, z, i_membership[(z * i_rows * i_cols) + y * i_rows + x]);
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

    // cuda streams
    cudaStream_t stream0, stream1, stream2;
	cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // allocate host locked memory for streams
    float *h_image, *h_membership, *h_cluster_centers;
    cudaHostAlloc((void **)&h_image, rows * cols * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_membership, rows * cols * i_num_clutsers * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_cluster_centers, i_num_clutsers * sizeof(float), cudaHostAllocDefault);

    // memcpy host locked memory with input values
    memcpy(h_image, i_image, rows * cols * sizeof(float));
    memcpy(h_membership, i_membership, rows * cols * i_num_clutsers * sizeof(float));
    memcpy(h_cluster_centers, i_cluster_centers, i_num_clutsers * sizeof(float));

    // cudaMemcpy everything!
    float *d_image, *d_membership, *d_cluster_centers;
    cudaMalloc((void**)&d_image, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_membership, rows * cols * i_num_clutsers * sizeof(float));
    cudaMalloc((void**)&d_cluster_centers, i_num_clutsers * sizeof(float));
    // cudaMalloc((void**)&out_membership, rows * cols * i_num_clutsers * sizeof(float));

    // ********** LEAVE THIS **********
    // cudaMemcpy(d_image, i_image, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_membership, i_membership, rows * cols * i_num_clutsers * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_cluster_centers, i_cluster_centers, i_num_clutsers * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_image, h_image, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_membership, h_membership, rows * cols * i_num_clutsers * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_centers, h_cluster_centers, i_num_clutsers * sizeof(float), cudaMemcpyHostToDevice);


    // kernel config parameters for reduction
    unsigned int blks_c = 1 + (rows * cols - 1) / threads_per_block;
    unsigned int in_size = rows * cols;
    unsigned int shared_size_c = threads_per_block * sizeof(float);

    // memory allocations for reduction
    float *numerator = new float[1];
    float *denominator = new float[1];
    float *device_in_num, *device_in_den, *device_out_num, *device_out_den;
    cudaMalloc((void **)&device_in_num, in_size * sizeof(float));
    cudaMalloc((void **)&device_in_den, in_size * sizeof(float));
    cudaMalloc((void **)&device_out_num, blks_c * sizeof(float));
    cudaMalloc((void **)&device_out_den, blks_c * sizeof(float));


    // iteratively update membership and centers for T epochs
    for (int i = 0; i < T; ++i) {
        // cout << "Hi" << endl;
        // fcm_step_kernel<<<dimGrid, dimBlock, shared_size>>>(i_image, i_membership, i_cluster_centers, rows, cols, i_num_clutsers, i_m);
        // 1D grid of 2D blocks
        update_membership_kernel<<<blks_m, threads_per_block, shared_size_m, stream0>>>(d_image, d_cluster_centers, d_membership, rows, cols, i_num_clutsers, i_m);
        // cudaDeviceSynchronize();
        // reduce each center with numerator and denominator reduction
        
        cudaMemcpyAsync(device_in_num, h_membership, in_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(device_in_den, h_membership, in_size * sizeof(float), cudaMemcpyHostToDevice, stream2);
        cudaDeviceSynchronize();
        for (int i = 0; i < i_num_clutsers; ++i) {
            blks_c = 1 + (rows * cols - 1) / threads_per_block;
            while (true) {
                update_centers_numerator_kernel<<<blks_c, threads_per_block, shared_size_c>>>(d_image, d_membership, d_cluster_centers, rows, cols, i_num_clutsers, i_m, device_out_num, i);
                update_centers_denominator_kernel<<<blks_c, threads_per_block, shared_size_c>>>(d_image, d_membership, d_cluster_centers, rows, cols, i_num_clutsers, i_m, device_out_den, i);
                // cudaDeviceSynchronize();
                // don't re-initialize if we already have the final reduction result
                if (blks_c == 1) {
                    break;
                }

                in_size = blks_c;

                // device_out now holds the reduce result, copy to device in and reduce again (next time)
                cudaMemcpy(device_in_num, device_out_num, in_size * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(device_in_den, device_out_den, in_size * sizeof(float), cudaMemcpyDeviceToDevice);
                // cudaDeviceSynchronize();

                blks_c = 1 + ((blks_c - 1) / threads_per_block);

            }
            cudaMemcpy(numerator, device_out_num, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(denominator, device_out_den, sizeof(float), cudaMemcpyDeviceToHost);

            cudaDeviceSynchronize();

            // update center values
            i_cluster_centers[i] = numerator[0] / denominator[0];
        }

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
    // cout << "Membership: " << endl;
    // for (int i = 0; i < i_rows; ++i) {
    //     for (int j = 0; j < i_cols; ++j) {
    //         float tmp_max = -999;
    //         for (int k = 0; k < i_num_clutsers; ++k) {
    //             // FIX!!
    //             // if (i_membership[i][j][k] >= tmp_max) {
    //             //     // cout << "hi" << endl;
    //             //     tmp_max = i_membership[i][j][k];
    //             //     // i_final_cluster[i][j] = 255 / (float)(k + 1);
    //             //     i_final_cluster[i][j] = k;
    //             // }
    //             int k = 0;
    //             // cout << i_membership[i][j][k] << " ";
    //         }
    //         // cout << endl;
    //     }
    // }

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

