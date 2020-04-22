#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/core.hpp"
#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
// #include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/highgui.hpp"
#include <iostream>
#include <cmath>
#include <chrono>
#include <ratio>
#include "sobel.h"

#include <omp.h>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    Mat image, norm_image;
    image = imread("license.jpg", 0); 	// Read the file
    if(image.empty())                      		// Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    
    normalize(image, norm_image, 0, 1, NORM_MINMAX, CV_32F);
    float *img = norm_image.ptr<float>(0);

    cout << image.rows << " " << image.cols << endl;

    // for (int i = 0; i < norm_image.rows; i++) {
    //     for (int j = 0; j < norm_image.cols; j++) {
    //         cout << img[i*image.cols+j] << " ";
    //     }
    //     cout << endl;
    // }

    chrono::high_resolution_clock::time_point start;
  	chrono::high_resolution_clock::time_point end;
  	chrono::duration<double, milli> duration_sec;

    float maskx[9] = {-1,-2,-1,0,0,0,1,2,1};
    float masky[9] = {-1,0,1,-2,0,2,-1,0,1};

    float maskx1[3] = {1, 0, -1};
    float maskx2[3] = {1, 2, 1};

    float masky1[3] = {1, 2, 1};
    float masky2[3] = {1, 0, -1};

    float* outputx = new float[image.rows * image.cols * 1];
    float* outputy = new float[image.rows * image.cols * 1];
    float* output = new float[image.rows * image.cols];

    float *temp = new float[image.rows * image.cols * 1];

    // omp_set_num_threads(atoi(argv[1]));
  	// start = chrono::high_resolution_clock::now();
    //  Convolve(img, outputx, image.rows, image.cols, maskx, 3);
    //  Convolve(img, outputy, image.rows, image.cols, masky, 3);
    //  end = chrono::high_resolution_clock::now();
    //  duration_sec = chrono::duration_cast<chrono::duration<double, milli>>(end - start);

    // omp_set_num_threads(atoi(argv[1]));

	// for (int i = 0; i < 3; i++) {

	// 	convolve1D_horiz(img, temp, image.rows, image.cols, maskx1, 3);
	//     convolve1D_vert(temp, outputx, image.rows, image.cols, maskx2, 3);

	//     convolve1D_horiz(img, temp, image.rows, image.cols, masky1, 3);
	//     convolve1D_vert(temp, outputy, image.rows, image.cols, masky2, 3);	
	
	// }
  	
  	start = chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(atoi(argv[1]))
    {
  	    // for (int i = 0; i < 10; i++) {

	   	   convolve1D_horiz(img, temp, image.rows, image.cols, maskx1, 3);
           // for (int i = 0; i < image.rows * image.cols; i++)
           //      temp[i] = temp[i*16];
	       convolve1D_vert(temp, outputx, image.rows, image.cols, maskx2, 3);
           // for (int i = 0; i < image.rows * image.cols; i++)
           //      outputx[i] = outputx[i*16];

	       convolve1D_horiz(img, temp, image.rows, image.cols, masky1, 3);
           // for (int i = 0; i < image.rows * image.cols; i++)
           //      temp[i] = temp[i*16];
           convolve1D_vert(temp, outputy, image.rows, image.cols, masky2, 3);
           // for (int i = 0; i < image.rows * image.cols; i++)
           //      outputy[i] = outputy[i*16];

  	    // }    
    
    
   
    
        //   omp_set_num_threads(atoi(argv[1]));

    	// for (int i = 0; i < 3; i++) {

    	// 	convolve1D_horiz_opt(img, temp, image.rows, image.cols, maskx1, 3);
    	//     convolve1D_vert_opt(temp, outputx, image.rows, image.cols, maskx2, 3);

    	//     convolve1D_horiz_opt(img, temp, image.rows, image.cols, masky1, 3);
    	//     convolve1D_vert_opt(temp, outputy, image.rows, image.cols, masky2, 3);	
    	
    	// }
  	
        // start = chrono::high_resolution_clock::now();

        // for (int i = 0; i < 10; i++) {

        //     convolve1D_horiz_opt(img, temp, image.rows, image.cols, maskx1, 3);
    	//     convolve1D_vert_opt(temp, outputx, image.rows, image.cols, maskx2, 3);

    	//     convolve1D_horiz_opt(img, temp, image.rows, image.cols, masky1, 3);
    	//     convolve1D_vert_opt(temp, outputy, image.rows, image.cols, masky2, 3);

        // }    
        
        // end = chrono::high_resolution_clock::now();
        // duration_sec = chrono::duration_cast<chrono::duration<double, milli>>(end - start);    

        #pragma omp for simd collapse(2)
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                output[i*image.cols+j] = sqrt(outputx[i*image.cols+j]*outputx[i*image.cols+j] + outputy[i*image.cols+j]*outputy[i*image.cols+j]);
            }
        }

    }

    end = chrono::high_resolution_clock::now();
    duration_sec = chrono::duration_cast<chrono::duration<double, milli>>(end - start);

    Mat out = Mat(image.rows, image.cols, CV_32F, output);
    Mat norm_out;
    normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_32F);    

    // for (int i = 0; i < out.rows; i++) {
    //     for (int j = 0; j < out.cols; j++) {
    //         cout << outputx[i*out.cols+j] << " ";
    //     }
    //     cout << endl;
    // }

    cout << duration_sec.count() << endl;

    Mat write_out;
    normalize(norm_out, write_out, 0, 255, NORM_MINMAX, CV_8U);
    imwrite("sobel1.png", write_out);  

    delete[] outputx;
    delete[] outputy;
    delete[] output;
    delete[] temp; 

    return 0;
}
