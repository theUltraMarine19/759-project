#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "fcm.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image, norm_image;
    image = imread( argv[1], 0);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    normalize(image, norm_image, 0, 1, NORM_MINMAX, CV_32F);
    
    int rows = image.rows, cols = image.cols;
    int num_clusters = 4, m = 2, epochs = 100;
    float epsilon = 0.05, mem_diff = 0;
    int **final_membership = new int* [rows];
    for (int i = 0; i < rows; ++i) {
        final_membership[i] = new int[cols];
        for (int j = 0; j < cols; ++j) {
            final_membership[i][j] = 255;
        }
    }

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


    FCM fcm(img, epsilon, rows, cols, num_clusters, m, final_membership);

    fcm.init_membership();
    fcm.init_centers();

    // mem_diff = fcm.update_membership();
    // fcm.update_centers();
    // mem_diff = fcm.update_membership();
    // fcm.update_centers();

    // run for 100 epochs
    for (int i = 0; i < epochs; ++i) {
        mem_diff = fcm.update_membership();
        fcm.update_centers();
    }


    cout << "400 Iterations done" << endl;


    // repaint image with corresponding cluster (255 / num_clusters)
    // fcm.print_mebership();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << final_membership[i][j] << " ";
        }
        // cout << "Yeehaw" << endl;
    }

    // normalize image
    Mat out = Mat(image.rows, image.cols, CV_8U, final_membership);
    // Mat norm_out;
    // normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_8U);

    Mat write_out;
    normalize(out, write_out, 0, 255, NORM_MINMAX, CV_8U);

    // show image
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", write_out);
    waitKey(0);

    cout << image.size() << endl;

    // namedWindow("Display Image", WINDOW_AUTOSIZE );
    // imshow("Display Image", image);
    // waitKey(0);


    return 0;
}
