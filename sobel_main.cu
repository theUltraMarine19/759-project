#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/core.hpp"
#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
#include <iostream>
#include "sobel.cuh"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
	
	int bdx = atoi(argv[1]);
	int bdy = atoi(argv[2]);
	
	cudaError_t err;

	// int dev;
	// cudaDeviceProp prop;
	// cudaGetDevice(&dev);
	// cudaGetDeviceProperties(&prop, dev);
	// cout << prop.sharedMemPerBlock << endl;
	
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	Mat image, norm_image;
    image = imread("license.jpg", 0); 	// Read the file
    if(image.empty())                      		// Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    
    normalize(image, norm_image, 0, 1, NORM_MINMAX, CV_32F);
    float *img = norm_image.ptr<float>(0);

    // for (int i = 0; i < 6; i++) {
    // 	for (int j = 0; j < 6; j++) {
    // 		cout << img[i*image.cols + j] << " ";
    // 	}
    // 	cout << endl;
    // }

    cout << image.rows << " " << image.cols << endl;

    float maskx[9] = {-1,-2,-1,0,0,0,1,2,1};
    float masky[9] = {-1,0,1,-2,0,2,-1,0,1};

    float maskx1[3] = {1, 0, -1};
    float maskx2[3] = {1, 2, 1};

    float masky1[3] = {1, 2, 1};
    float masky2[3] = {1, 0, -1};

    float *dimg, *outx, *outy, *output;
    err = cudaMalloc((void **)&dimg, image.rows * image.cols * sizeof(float));
	  // cout << cudaGetErrorName(err) << endl;
  	err = cudaMallocManaged((void **)&outx, image.rows * image.cols * sizeof(float));
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaMallocManaged((void **)&outy, image.rows * image.cols * sizeof(float));
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaMallocManaged((void **)&output, image.rows * image.cols * sizeof(float));
  	// cout << cudaGetErrorName(err) << endl;
  	
  	float *dmaskx, *dmasky, *dmaskx1, *dmaskx2; // masky1 = maskx2 and masky2 = maskx1
  	err = cudaMalloc((void **)&dmaskx, 9 * sizeof(float));
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaMalloc((void **)&dmasky, 9 * sizeof(float));
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaMalloc((void **)&dmaskx1, 3 * sizeof(float));
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaMalloc((void **)&dmaskx2, 3 * sizeof(float));
	// cout << cudaGetErrorName(err) << endl;
  	
  	
  	err = cudaMemcpy(dimg, norm_image.ptr<float>(), image.rows * image.cols * sizeof(float), cudaMemcpyHostToDevice);
  	cout << cudaGetErrorName(err) << endl;
  	err = cudaMemcpy(dmaskx, maskx, 9 * sizeof(float), cudaMemcpyHostToDevice);
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaMemcpy(dmasky, masky, 9 * sizeof(float), cudaMemcpyHostToDevice);
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaMemcpy(dmaskx1, maskx1, 3 * sizeof(float), cudaMemcpyHostToDevice);
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaMemcpy(dmaskx2, maskx2, 3 * sizeof(float), cudaMemcpyHostToDevice);
  	// cout << cudaGetErrorName(err) << endl;
  	
  	// Can be improved with CUDA streams
  	cudaEventRecord(start);
  	conv(dimg, dmaskx, outx, image.rows, image.cols, bdx, bdy);
  	conv(dimg, dmasky, outy, image.rows, image.cols, bdx, bdy);

    // conv_opt(dimg, dmaskx1, dmaskx2, outx, image.rows, image.cols, bdx, bdy);
    // conv_opt(dimg, dmaskx2, dmaskx1, outy, image.rows, image.cols, bdx, bdy);
     	
   	dim3 block(bdx, bdy);
  	dim3 grid((image.cols + block.x - 1) / block.x, (image.rows + block.y - 1) / block.y);
   	magnitude<<<grid, block>>>(outx, outy, output, image.rows, image.cols);
   	err = cudaDeviceSynchronize();
  	// cout << cudaGetErrorName(err) << endl;
  	
  	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);

  	float ms;
  	cudaEventElapsedTime(&ms, start, stop);

  	Mat out = Mat(image.rows, image.cols, CV_32F, output);
    Mat norm_out;
    normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_32F);

  	cout << ms << endl;

  	Mat write_out;
    normalize(norm_out, write_out, 0, 255, NORM_MINMAX, CV_8U);
    imwrite("sobel1_CUDA.png", write_out);

  	err = cudaFree(dimg);
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaFree(outx);
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaFree(outy);
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaFree(output);
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaFree(dmaskx);
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaFree(dmasky);
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaFree(dmaskx1);
  	// cout << cudaGetErrorName(err) << endl;
  	err = cudaFree(dmaskx2);
  	// cout << cudaGetErrorName(err) << endl;
  	
  	cout << "Done!\n";
  	
  	return 0;
}
