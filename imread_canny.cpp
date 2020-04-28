#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/core.hpp"
#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
// #include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/highgui.hpp"
#include <iostream>
#include <cmath>
#include <chrono>
#include <ratio>

#include <omp.h>
#include "sobel.h"
#include "canny.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    Mat image, norm_image;
    image = imread("license.jpg", 0); // Read the file
    if(image.empty())                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    /* Normalize the input image to [0, 1] */
    normalize(image, norm_image, 0.0, 1.0, NORM_MINMAX, CV_32F);
    float *img = norm_image.ptr<float>(0);
    
    // cout << image.rows << " " << image.cols << endl;

    // for (int i = 0; i < norm_image.rows; i++) {
    //     for (int j = 0; j < norm_image.cols; j++) {
    //         cout << img[i*image.cols+j] << " ";
    //     }
    //     cout << endl;
    // }

    chrono::high_resolution_clock::time_point start;
    chrono::high_resolution_clock::time_point end;
    chrono::duration<double, milli> duration_sec;

    float *temp = new float[image.rows * image.cols];
    float *smooth_img = new float[image.rows * image.cols];

    float maskx1[3] = {1, 0, -1};
    float maskx2[3] = {1, 2, 1};

    float masky1[3] = {1, 2, 1};
    float masky2[3] = {1, 0, -1};

    float* outputx = new float[image.rows * image.cols];
    float* outputy = new float[image.rows * image.cols];
    
    float* output = new float[image.rows * image.cols];
    float* grad = new float[image.rows * image.cols];
    
    float* supp = new float[image.rows * image.cols];

    float filter[9];
    
    omp_set_num_threads(atoi(argv[1]));

    start = chrono::high_resolution_clock::now();
    generateGaussian(filter, 3, 1.0);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cout << filter[i*3+j] << " ";
        }
        cout << endl;
    }

    /* Do a convolution of image with the Gaussian filter */
    Convolve(img, smooth_img, image.rows, image.cols, filter, 3);
    
    Mat smoothed = Mat(image.rows, image.cols, CV_32F, smooth_img);
    Mat norm_smoothed;
    normalize(smoothed, norm_smoothed, 0, 1, NORM_MINMAX, CV_32F);
    delete[] smooth_img;
    smooth_img = norm_smoothed.ptr<float>(0);

    /* get the output Gx, by convolution with Sobel operator */
    convolve1D_horiz(smooth_img, temp, image.rows, image.cols, maskx1, 3);
    convolve1D_vert(temp, outputx, image.rows, image.cols, maskx2, 3);

    /* get the output Gy, by convolution with Sobel operator */
    convolve1D_horiz(smooth_img, temp, image.rows, image.cols, masky1, 3);
    convolve1D_vert(temp, outputy, image.rows, image.cols, masky2, 3);

    mag_grad(outputx, outputy, output, grad, image.rows, image.cols);
    
    Mat mag = Mat(image.rows, image.cols, CV_32F, output);
    Mat norm_mag;
    normalize(mag, norm_mag, 0, 1, NORM_MINMAX, CV_32F);
    delete[] output;
    output = norm_mag.ptr<float>(0);

    memcpy(supp, output, image.rows * image.cols * sizeof(float));
    NonMaxSuppresion(grad, output, supp, image.rows, image.cols);

    Mat out = Mat(image.rows, image.cols, CV_32F, supp);
    Mat norm_out;
    normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_32F);
    float* norm_supp = norm_out.ptr<float>(0);

    // double minVal, maxVal;
    // minMaxLoc(norm_out, &minVal, &maxVal);
    // cout << minVal << " " << maxVal << endl;

    // for (int i = 0; i < out.rows; i++) {
    //     for (int j = 0; j < out.cols; j++) {
    //         // cout << supp[i*out.cols+j] << " ";
    //     }
    //     cout << endl;
    // }

    q_hysteresis(norm_supp, image.rows, image.cols, 0.08, 0.11);

    // for (int i = 0; i < norm_out.rows; i++) {
    //     for (int j = 0; j < 10; j++) {
    //         cout << norm_supp[i*out.cols+j] << " ";
    //     }
    //     cout << endl;
    // }

    end = chrono::high_resolution_clock::now();
    duration_sec = chrono::duration_cast<chrono::duration<double, milli>>(end - start);

    out = Mat(image.rows, image.cols, CV_32F, norm_supp);
    normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_32F);

    cout << duration_sec.count() << endl;

    Mat write_out;
    normalize(norm_out, write_out, 0, 255, NORM_MINMAX, CV_8U);
    imwrite("canny2.png", write_out);

    delete[] temp;
    delete[] outputx;
    delete[] outputy;
    delete[] grad;
    delete[] supp;

    return 0;
}
