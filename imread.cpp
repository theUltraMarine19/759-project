#include "/home/arijit/installation/OpenCV-3.4.4/include/opencv2/core.hpp"
#include "/home/arijit/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
#include "/home/arijit/installation/OpenCV-3.4.4/include/opencv2/highgui.hpp"
#include <iostream>
#include <cmath>
#include "sobel.h"

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
    imshow( "Display window", image);
    waitKey(0);
    
    normalize(image, norm_image, 0, 1, NORM_MINMAX, CV_32F);
    float *img = norm_image.ptr<float>(0);

    imshow( "Display window", norm_image);
    waitKey(0);
    cout << image.rows << " " << image.cols << endl;

    // for (int i = 0; i < norm_image.rows; i++) {
    //     for (int j = 0; j < norm_image.cols; j++) {
    //         cout << img[i*image.cols+j] << " ";
    //     }
    //     cout << endl;
    // }

    float maskx[9] = {-1,-2,-1,0,0,0,1,2,1};
    float masky[9] = {-1,0,1,-2,0,2,-1,0,1};

    float maskx1[3] = {1, 0, -1};
    float maskx2[3] = {1, 2, 1};

    float masky1[3] = {1, 2, 1};
    float masky2[3] = {1, 0, -1};

    float* outputx = new float[image.rows * image.cols];
    float* outputy = new float[image.rows * image.cols];
    float* output = new float[image.rows * image.cols];

    float *temp = new float[image.rows * image.cols];

    // Convolve(img, outputx, image.rows, image.cols, maskx, 3);
    // Convolve(img, outputy, image.rows, image.cols, masky, 3);

    convolve1D_horiz(img, temp, image.rows, image.cols, maskx1, 3);
    convolve1D_vert(temp, outputx, image.rows, image.cols, maskx2, 3);

    convolve1D_horiz(img, temp, image.rows, image.cols, masky1, 3);
    convolve1D_vert(temp, outputy, image.rows, image.cols, masky2, 3);    

    #pragma omp for simd collapse(2)
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            output[i*image.cols+j] = sqrt(outputx[i*image.cols+j]*outputx[i*image.cols+j] + outputy[i*image.cols+j]*outputy[i*image.cols+j]);
        }
    }

    Mat out = Mat(image.rows, image.cols, CV_32F, output);
    Mat norm_out;
    normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_32F);    

    // for (int i = 0; i < out.rows; i++) {
    //     for (int j = 0; j < out.cols; j++) {
    //         cout << outputx[i*out.cols+j] << " ";
    //     }
    //     cout << endl;
    // }    

    imshow( "Display window", norm_out);
    waitKey(0);
    return 0;
}
