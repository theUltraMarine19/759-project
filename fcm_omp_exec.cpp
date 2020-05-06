#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <ratio>
#include <cmath>
#include "/srv/home/bchang/installation/OpenCV-3.4.4/include/opencv2/core/core.hpp"
#include "/srv/home/bchang/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
// #include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
#include "fcm.h"

using namespace cv;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char** argv)
{
    std::cout << "Start" << endl;
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image, norm_image;
    std::cout << "Start?" << endl;
    image = imread( argv[1], 0);
    std::cout << "Start??" << endl;
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    // std::cout << "Start?????" << endl;
    normalize(image, norm_image, 0, 1, NORM_MINMAX, CV_32F);
    // std::cout << "dafuq?" << endl;

    // std::cout << "222111?" << endl;
    
    int rows = image.rows, cols = image.cols;
    int num_clusters = 7, m = 2, epochs = 100;
    float epsilon = 0.05, mem_diff = 0;
    int **final_membership = new int* [rows];
    for (int i = 0; i < rows; ++i) {
        final_membership[i] = new int[cols];
        for (int j = 0; j < cols; ++j) {
            final_membership[i][j] = -1;
        }
    }
    // std::cout << "222?" << endl;

    float **img = new float*[rows];

    for (int i = 0; i < rows; ++i) {
        img[i] = new float[cols];
    }

    for (int i = 0; i < rows; ++i) {
        img[i] = norm_image.ptr<float>(i);;
    }

    // for (int i = 0; i < rows; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //         cout << img[i][j] << " ";
    //     }
    //     cout <<endl;
    // }

    // std::cout << "333?" << endl;
    FCM fcm(img, epsilon, rows, cols, num_clusters, m, final_membership);
    // std::cout << "444?" << endl;

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    start = high_resolution_clock::now();
    fcm.init_membership();
    fcm.init_centers();
    // run for 100 epochs
    for (int i = 0; i < epochs; ++i) {
        // std::cout << "666?" << endl;
        mem_diff = fcm.update_membership();
        fcm.update_centers();
    }
    end = high_resolution_clock::now();
    


    std::cout << "100 Iterations done" << endl;


    // repaint image with corresponding cluster (255 / num_clusters)
    fcm.print_mebership();

    // for (int i = 0; i < rows; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //         cout << final_membership[i][j] << " ";
    //     }
    //     // cout << "Yeehaw" << endl;
    // }
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << duration_sec.count()<<endl;



    std::cout << "Saving values..." << endl;
    ofstream myfile ("output.txt");
    if (myfile.is_open()) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                myfile << final_membership[i][j] << " ";
            }
            myfile << "\n";
        }
        myfile.close();
    }

     // normalize image
    Mat out = Mat(image.rows, image.cols, CV_32F, final_membership);
    Mat norm_out;
    normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_32F);
    Mat write_out;
    normalize(norm_out, write_out, 0, 255, NORM_MINMAX, CV_8U);

    // show image
    // namedWindow("Display Image", WINDOW_AUTOSIZE );
    // imshow("Display Image", write_out);
    // waitKey(0);
    std::cout << "Done" << endl;



    cout << image.size() << endl;

    // namedWindow("Display Image", WINDOW_AUTOSIZE );
    // imshow("Display Image", image);
    // waitKey(0);


    return 0;
}
