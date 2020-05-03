#include <stdio.h>
#include <iostream>
#include <fstream>
#include "/srv/home/bchang/installation/OpenCV-3.4.4/include/opencv2/core/core.hpp"
#include "/srv/home/bchang/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
// #include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
#include "fcm.h"

using namespace cv;
using namespace std;

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
    int num_clusters = 6, m = 2, epochs = 50;
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

    std::cout << "333?" << endl;
    FCM fcm(img, epsilon, rows, cols, num_clusters, m, final_membership);
    std::cout << "444?" << endl;

    fcm.init_membership();
    fcm.init_centers();
    std::cout << "555?" << endl;

    // mem_diff = fcm.update_membership();
    // fcm.update_centers();
    // mem_diff = fcm.update_membership();
    // fcm.update_centers();

    // run for 100 epochs
    for (int i = 0; i < epochs; ++i) {
        // std::cout << "666?" << endl;
        mem_diff = fcm.update_membership();
        fcm.update_centers();
    }


    std::cout << "400 Iterations done" << endl;


    // repaint image with corresponding cluster (255 / num_clusters)
    fcm.print_mebership();

    // for (int i = 0; i < rows; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //         cout << final_membership[i][j] << " ";
    //     }
    //     // cout << "Yeehaw" << endl;
    // }


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
