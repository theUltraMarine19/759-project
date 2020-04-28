#include <stdio.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include "/srv/home/bchang/installation/OpenCV-3.4.4/include/opencv2/core/core.hpp"
#include "/srv/home/bchang/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
#include "fcm_cuda.cuh"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout<< "Wrong number of arguments (should take 2, got " << argc - 1 << ")" << std::endl;
        return 0;
    }

    // set hyperparameters
    int num_clusters = 6, m = 2, epochs = 100;

    // variables for timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    // threads per block
    unsigned int block_dim = atoi(argv[1]);

    // variables for computation
    float ***i_membership;
    float *i_cluster_centers;
    float i_terminate_epsilon;
    int **i_final_cluster;
    int i_rows, i_cols;
    int i_num_clutsers;
    int i_image_size;
    int i_m;

    // read image from cmd line
    Mat image, norm_image;
    image = imread( argv[2], 0);

    int rows = image.rows, cols = image.cols;
    float epsilon = 0.05, mem_diff = 0;
    int **final_membership = new int* [rows];
    for (int i = 0; i < rows; ++i) {
        final_membership[i] = new int[cols];
        for (int j = 0; j < cols; ++j) {
            final_membership[i][j] = -1;
        }
    }

    // load image into array
    float **img = new float*[rows];
    for (int i = 0; i < rows; ++i) {
        img[i] = new float[cols];
    }
    for (int i = 0; i < rows; ++i) {
        img[i] = norm_image.ptr<float>(i);;
    }

    // initialize variables for computation
    init_membership(i_membership);
    init_centers(i_cluster_centers);


    // perform FCM and measure execution time here
    cudaEventRecord(start);
    // RUN_FCM_HERE
    fcm_step(i_image, i_membership, rows, cols, epochs, num_clusters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::printf("%f\n", ms);


    return 0;
}