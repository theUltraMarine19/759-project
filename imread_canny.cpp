#include "/home/arijit/installation/OpenCV-3.4.4/include/opencv2/core.hpp"
#include "/home/arijit/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
#include "/home/arijit/installation/OpenCV-3.4.4/include/opencv2/highgui.hpp"
#include <iostream>
#include <cmath>
#include "sobel.h"
#include "canny.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    Mat image, norm_image;
    image = imread("Swimming-club.jpg", 0); // Read the file
    if(image.empty())                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "Display window", WINDOW_AUTOSIZE );
    
    // imshow( "Display window", image);
    // waitKey(0);
    
    /* Normalize the input image to [0, 1] */
    normalize(image, norm_image, 0.0, 1.0, NORM_MINMAX, CV_32F);
    float *img = norm_image.ptr<float>(0);
    
    // imshow( "Display window", norm_image);
    // waitKey(0);
    
    // cout << image.rows << " " << image.cols << endl;

    // for (int i = 0; i < norm_image.rows; i++) {
    //     for (int j = 0; j < norm_image.cols; j++) {
    //         cout << img[i*image.cols+j] << " ";
    //     }
    //     cout << endl;
    // }

    float *temp = new float[image.rows * image.cols];
    float *smooth_img = new float[image.rows * image.cols];

    float filter[9];
    generateGaussian(filter, 3, 1.0);

    // for (int i = 0; i < 3; i++) {
    //     for (int j = 0; j < 3; j++) {
    //         cout << filter[i*3+j] << " ";
    //     }
    //     cout << endl;
    // }

    /* Do a convolution of image with the Gaussian filter */
    Convolve(img, smooth_img, image.rows, image.cols, filter, 3);
    Mat smoothed = Mat(image.rows, image.cols, CV_32F, smooth_img);
    
    // imshow( "Display window", smoothed);
    // waitKey(0);

    Mat norm_smoothed;
    normalize(smoothed, norm_smoothed, 0, 1, NORM_MINMAX, CV_32F);
    
    // imshow( "Display window", norm_smoothed);
    // waitKey(0);    

    smooth_img = norm_smoothed.ptr<float>(0);

    float maskx1[3] = {1, 0, -1};
    float maskx2[3] = {1, 2, 1};

    float masky1[3] = {1, 2, 1};
    float masky2[3] = {1, 0, -1};

    float* outputx = new float[image.rows * image.cols];
    float* outputy = new float[image.rows * image.cols];
    
    float* output = new float[image.rows * image.cols];
    float* grad = new float[image.rows * image.cols];
    
    float* supp = new float[image.rows * image.cols];

    /* get the output Gx, by convolution with Sobel operator */
    convolve1D_horiz(smooth_img, temp, image.rows, image.cols, maskx1, 3);
    convolve1D_vert(temp, outputx, image.rows, image.cols, maskx2, 3);

    /* get the output Gy, by convolution with Sobel operator */
    convolve1D_horiz(smooth_img, temp, image.rows, image.cols, masky1, 3);
    convolve1D_vert(temp, outputy, image.rows, image.cols, masky2, 3);

    // Mat ox = Mat(image.rows, image.cols, CV_32F, outputx);
    // Mat oy = Mat(image.rows, image.cols, CV_32F, outputy);
    // Mat norm_ox, norm_oy;
    // normalize(ox, norm_ox, 0, 1, NORM_MINMAX, CV_32F);
    // normalize(oy, norm_oy, 0, 1, NORM_MINMAX, CV_32F);
    
    // imshow( "Display window", ox);
    // waitKey(0);
    // imshow( "Display window", oy);
    // waitKey(0);

    // imshow( "Display window", norm_ox);
    // waitKey(0);
    // imshow( "Display window", norm_oy);
    // waitKey(0);

    // outputx = ox.ptr<float>(0);
    // outputy = oy.ptr<float>(0);

    // #pragma omp for simd collapse(2)
    // for (int i = 0; i < image.rows; i++) {
    //     for (int j = 0; j < image.cols; j++) {
    //         output[i*image.cols+j] = sqrt(outputx[i*image.cols+j]*outputx[i*image.cols+j] + outputy[i*image.cols+j]*outputy[i*image.cols+j]);
    //     }
    // }

    mag_grad(outputx, outputy, output, grad, image.rows, image.cols);
    Mat mag = Mat(image.rows, image.cols, CV_32F, output);
    // imshow( "Display window", mag);
    // waitKey(0);

    Mat norm_mag;
    normalize(mag, norm_mag, 0, 1, NORM_MINMAX, CV_32F);
    // imshow( "Display window", norm_mag);
    // waitKey(0);

    output = norm_mag.ptr<float>(0);

    memcpy(supp, output, image.rows * image.cols * sizeof(float));
    NonMaxSuppresion(grad, output, supp, image.rows, image.cols);

    Mat out = Mat(image.rows, image.cols, CV_32F, supp);
    // imshow( "Display window", out);
    // waitKey(0);

    Mat norm_out;
    normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_32F);
    imshow( "Display window", norm_out);
    waitKey(0);    

    double minVal, maxVal;
    minMaxLoc(norm_out, &minVal, &maxVal);

    cout << minVal << " " << maxVal << endl;

    float* norm_supp = norm_out.ptr<float>(0);


    // for (int i = 0; i < out.rows; i++) {
    //     for (int j = 0; j < out.cols; j++) {
    //         // cout << supp[i*out.cols+j] << " ";
    //     }
    //     cout << endl;
    // }

    hysteresis(norm_supp, image.rows, image.cols, 0.08, 0.11);

    // for (int i = 0; i < norm_out.rows; i++) {
    //     for (int j = 0; j < 10; j++) {
    //         cout << norm_supp[i*out.cols+j] << " ";
    //     }
    //     cout << endl;
    // }

    out = Mat(image.rows, image.cols, CV_32F, norm_supp);
    // imshow( "Display window", out);
    // waitKey(0);

    normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_32F);
    imshow( "Display window", norm_out);
    waitKey(0);
    
    return 0;
}
