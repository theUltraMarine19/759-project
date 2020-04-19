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

    float **img = new float*[rows];

    for (int i = 0; i < rows; ++i) {
        img[i] = new float[cols];
    }

    for (int i = 0; i < rows; ++i) {
        img[i] = norm_image.ptr<float>(i);;
    }


    FCM fcm(img, epsilon, rows, cols, num_clusters, m);

    fcm.init_membership();
    fcm.init_centers();

    // run for 100 epochs
    for (int i = 0; i < epochs; ++i) {
        fcm.update_centers();
        fcm.update_membership();
    }

    // repaint image with corresponding cluster (255 / num_clusters)

    // normalize image

    // show image

    cout << image.size() << endl;

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);

    delete fcm;
    return 0;
}
