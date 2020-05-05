#include "/srv/home/bchang/installation/OpenCV-3.4.4/include/opencv2/core/core.hpp"
#include "/srv/home/bchang/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <stdlib.h>
#include <random>
#include "reduction_fcm_iter.cuh"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout<< "Wrong number of arguments (should take 2, got " << argc - 1 << ")" << std::endl;
        return 0;
    }

    // set hyperparameters
    int num_clusters = 7, i_m = 2, epochs = 200;

    // variables for timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    // threads per block
    unsigned int block_dim = atoi(argv[1]);


    // int *i_final_cluster;

    // read image from cmd line
    Mat image, norm_image;
    image = imread( argv[2], 0);
    normalize(image, norm_image, 0, 1, NORM_MINMAX, CV_32F);

    int rows = image.rows, cols = image.cols;
    // float epsilon = 0.05, mem_diff = 0;
    // int **final_membership = new int* [rows];
    // for (int i = 0; i < rows; ++i) {
    //     final_membership[i] = new int[cols];
    //     for (int j = 0; j < cols; ++j) {
    //         final_membership[i][j] = -1;
    //     }
    // }

    float **img2 = new float*[rows];

    for (int i = 0; i < rows; ++i) {
        img2[i] = new float[cols];
    }

    for (int i = 0; i < rows; ++i) {
        img2[i] = norm_image.ptr<float>(i);;
    }

    // float *img = norm_image.ptr<float>(0);
    float *img = new float[rows * cols];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            img[i + j * rows] = img2[i][j];
        }
    }

    // load image into array
    // float **img = new float*[rows];
    // for (int i = 0; i < rows; ++i) {
    //     img[i] = new float[cols];
    // }
    // for (int i = 0; i < rows; ++i) {
    //     img[i] = norm_image.ptr<float>(i);;
    // }

    cout << "Rows: " << rows << endl;
    cout << "Cols: " << cols << endl;

    // initialize variables for computation
    // init_membership(i_membership, rows, cols, num_clusters);
    // init_centers(i_cluster_centers, num_clusters);
    
    // variables for computation
    float *i_membership = new float[rows * cols * num_clusters];
    float *i_cluster_centers = new float[num_clusters];
    for (int i = 0; i < rows * cols * num_clusters; ++i) {
        i_membership[i] = 1 / (float)num_clusters;
    }

    cout << "Mem: " << 1 / (float)num_clusters << endl;

    for (int i = 0; i < num_clusters; ++i) {
        // randomly select i_num_clutsers points as cluster centers

        // random generator
        random_device rd;
        mt19937 eng(rd());
        uniform_real_distribution<> dist(0, 1);

        i_cluster_centers[i] = dist(eng);
        // i_cluster_centers[i] = (i + 1) / 10;
    }





    // perform FCM and measure execution time here
    float *out_membership = new float[num_clusters * rows * cols];
    int *final_cluster = new int[rows * cols];
    init_final_cluster(final_cluster, rows, cols);
    cudaEventRecord(start);
    // RUN_FCM_HERE
    fcm_step(img, i_membership, i_cluster_centers, rows, cols, epochs, num_clusters, i_m, block_dim, out_membership);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    calculate_final_cluster(out_membership, final_cluster, num_clusters, rows, cols);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::printf("%f\n", ms);
    

    cout << "Saving values..." << endl;
    ofstream myfile ("output.txt");
    if (myfile.is_open()) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                myfile << final_cluster[i + j * rows] << " ";
            }
            myfile << "\n";
        }
        myfile.close();
    }


    return 0;
}