#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/core.hpp"
#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
#include <iostream>
#include <queue>
#include <omp.h>
#include "canny.cuh"
#include "sobel.cuh"
#include "canny.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
	
  	int bdx = atoi(argv[1]);
  	int bdy = atoi(argv[2]);
    int t = atoi(argv[3]);
  	
  	cudaError_t err;

  	cudaEvent_t start;
  	cudaEvent_t stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);

  	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

    omp_set_num_threads(t);

  	Mat image, norm_image;
    image = imread("license.jpg", 0); 	
    if(image.empty())                   
    {
        cout <<  "Could not open or find the image" << endl;
        return -1;
    }
    
    normalize(image, norm_image, 0, 1, NORM_MINMAX, CV_32F);
    float *img = norm_image.ptr<float>(0);

    // cout << image.rows << " " << image.cols << endl;

    float maskx[9] = {-1,-2,-1,0,0,0,1,2,1};
    float masky[9] = {-1,0,1,-2,0,2,-1,0,1};

    float maskx1[3] = {1, 0, -1};
    float maskx2[3] = {1, 2, 1};

    float *filter, *dimg, *doutx, *douty, *doutput;
    
    err = cudaMalloc((void **)&dimg, image.rows * image.cols * sizeof(float));
    err = cudaMalloc((void **)&filter, 3 * 3 * sizeof(float));
	err = cudaMalloc((void **)&doutx, image.rows * image.cols * sizeof(float));
  	err = cudaMalloc((void **)&douty, image.rows * image.cols * sizeof(float));
  	err = cudaMalloc((void **)&doutput, image.rows * image.cols * sizeof(float));
	
	float *himg, *houtx, *houty, *houtput, *hfilter;
    // err = cudaHostAlloc((void **)&hfilter, 9*sizeof(float), cudaHostAllocDefault);
  	err = cudaHostAlloc((void **)&himg, image.rows*image.cols*sizeof(float), cudaHostAllocDefault);
  	err = cudaHostAlloc((void **)&houtx, image.rows*image.cols*sizeof(float), cudaHostAllocDefault);
  	err = cudaHostAlloc((void **)&houty, image.rows*image.cols*sizeof(float), cudaHostAllocDefault);
  	err = cudaHostAlloc((void **)&houtput, image.rows*image.cols*sizeof(float), cudaHostAllocDefault);

    memcpy(himg, norm_image.ptr<float>(), image.rows * image.cols * sizeof(float));

    float *dgrad;
    err = cudaMalloc((void **)&dgrad, image.rows * image.cols * sizeof(float));
    
    float *dmaskx, *dmasky, *dmaskx1, *dmaskx2; 			// masky1 = maskx2 and masky2 = maskx1
  	err = cudaMalloc((void **)&dmaskx, 9 * sizeof(float));
  	err = cudaMalloc((void **)&dmasky, 9 * sizeof(float));
  	err = cudaMalloc((void **)&dmaskx1, 3 * sizeof(float));
  	err = cudaMalloc((void **)&dmaskx2, 3 * sizeof(float));

  	float *hmaskx, *hmasky, *hmaskx1, *hmaskx2;
  	cudaHostAlloc((void **)&hmaskx, 9*sizeof(float), cudaHostAllocDefault);
  	cudaHostAlloc((void **)&hmasky, 9*sizeof(float), cudaHostAllocDefault);
  	cudaHostAlloc((void **)&hmaskx1, 3*sizeof(float), cudaHostAllocDefault);
  	cudaHostAlloc((void **)&hmaskx2, 3*sizeof(float), cudaHostAllocDefault);

  	memcpy(hmaskx, maskx, 9 * sizeof(float));
  	memcpy(hmasky, masky, 9 * sizeof(float));
  	memcpy(hmaskx1, maskx1, 3 * sizeof(float));
  	memcpy(hmaskx2, maskx2, 3 * sizeof(float));

  	
  	int* ctr;
	cudaMallocManaged((void **)&ctr, sizeof(int));
	cudaMemset(ctr, 0, sizeof(int));  	

	float *temp, *temp1;
	err = cudaMalloc((void **)&temp, image.rows * image.cols * sizeof(float));
	err = cudaMalloc((void **)&temp1, image.rows * image.cols * sizeof(float));

  	cudaEventRecord(start);

    dim3 block(3, 3);
    dim3 grid(1, 1);

    err = cudaMemcpyAsync(dimg, himg, image.rows * image.cols * sizeof(float), cudaMemcpyHostToDevice, stream1);
    // err = cudaMemcpyAsync(filter, hfilter, 9 * sizeof(float), cudaMemcpyHostToDevice, stream0);
    // generateGaussian(hfilter, 3, 1.0);

    generateGaussian<<<grid, block, 0, stream0>>>(filter, 1.0);
    
    // perform some CPU-side instructions before waiting for above 2 streams to finish
    block.x = bdx; block.y = bdy;
    grid.x = (image.cols + block.x - 1)/block.x; grid.y = (image.rows + block.y - 1)/block.y;
	err = cudaDeviceSynchronize();

	conv_kernel<<<grid, block, sizeof(float) * (bdx + 2) * (bdy + 2) + 3 * 3 * sizeof(float) + sizeof(float) * bdx * bdy, stream0>>>(dimg, filter, doutput, image.rows, image.cols);
	err = cudaMemcpyAsync(dmaskx1, hmaskx1, 3 * sizeof(float), cudaMemcpyHostToDevice, stream1);

	// cudaStreamSynchronize(stream0); // wait for conv_kernel to finish
	// err = cudaMemcpyAsync(himg, doutput, image.rows * image.cols * sizeof(float), cudaMemcpyDeviceToHost, stream0);

	err = cudaMemcpy(himg, doutput, image.rows * image.cols * sizeof(float), cudaMemcpyDeviceToHost);	
    
    Mat smoothed = Mat(image.rows, image.cols, CV_32F, himg);
    Mat norm_smoothed;
    normalize(smoothed, norm_smoothed, 0, 1, NORM_MINMAX, CV_32F);
    
    // No advantage of Async since have to wait for this to finish
    err = cudaMemcpy(dimg, norm_smoothed.ptr<float>(), image.rows * image.cols * sizeof(float), cudaMemcpyHostToDevice);
    
    conv_kernel_horiz<<<grid, block, sizeof(float) * bdy * (bdx+2) + 3 * sizeof(float) + sizeof(float) * bdx * bdy, stream1>>>(dimg, dmaskx1, temp, image.rows, image.cols);
  	err = cudaMemcpyAsync(dmaskx2, hmaskx2, 3 * sizeof(float), cudaMemcpyHostToDevice, stream0);
  	
  	err = cudaDeviceSynchronize();

  	conv_kernel_horiz<<<grid, block, sizeof(float) * bdy * (bdx+2) + 3 * sizeof(float) + sizeof(float) * bdx * bdy, stream0>>>(dimg, dmaskx2, temp1, image.rows, image.cols);
  	conv_kernel_vert<<<grid, block, sizeof(float) * (bdy+2) * bdx + 3 * sizeof(float) + sizeof(float) * bdx * bdy, stream1>>>(temp, dmaskx2, doutx, image.rows, image.cols);

  	conv_kernel_vert<<<grid, block, sizeof(float) * (bdy+2) * bdx + 3 * sizeof(float) + sizeof(float) * bdx * bdy, stream0>>>(temp1, dmaskx1, douty, image.rows, image.cols);
  	err = cudaMemcpyAsync(houtx, doutx, image.rows * image.cols * sizeof(float), cudaMemcpyDeviceToHost, stream1);

  	mag_grad<<<grid, block, 0, stream0>>>(doutx, douty, doutput, dgrad, image.rows, image.cols);
    err = cudaMemcpyAsync(houty, douty, image.rows * image.cols * sizeof(float), cudaMemcpyDeviceToHost, stream1);
  	
    err = cudaMemcpy(houtput, doutput, image.rows * image.cols * sizeof(float), cudaMemcpyDeviceToHost);

    Mat mag = Mat(image.rows, image.cols, CV_32F, houtput);
    Mat norm_mag;
    normalize(mag, norm_mag, 0, 1, NORM_MINMAX, CV_32F);
    

	err = cudaMemcpy(dimg, norm_mag.ptr<float>(), image.rows * image.cols * sizeof(float), cudaMemcpyHostToDevice);
	err = cudaMemcpy(doutput, dimg, image.rows * image.cols * sizeof(float), cudaMemcpyDeviceToDevice);
    
	NonMaxSuppression<<<grid, block, (bdx+2)*(bdy+2)*sizeof(float), stream0>>>(dgrad, dimg, doutput, image.rows, image.cols);
    
    err = cudaMemcpy(houtput, doutput, image.rows * image.cols * sizeof(float), cudaMemcpyDeviceToHost);

  	hysteresis(houtput, image.rows, image.cols, 0.08, 0.11);

  	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);

  	float ms;
  	cudaEventElapsedTime(&ms, start, stop);

  	Mat out = Mat(image.rows, image.cols, CV_32F, houtput);
    Mat norm_out;
    normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_32F);

  	cout << ms << endl;

	Mat write_out;
	normalize(norm_out, write_out, 0, 255, NORM_MINMAX, CV_8U);
	imwrite("canny1_opt.png", write_out);

	err = cudaFree(dimg);
  	err = cudaFree(filter);
    err = cudaFree(dgrad);
  	err = cudaFree(doutx);
  	err = cudaFree(douty);
  	err = cudaFree(doutput);
  	err = cudaFree(dmaskx);
  	err = cudaFree(dmasky);
  	err = cudaFree(dmaskx1);
  	err = cudaFree(dmaskx2);

  	err = cudaFreeHost(himg);
	// err = cudaFreeHost(hfilter);
	err = cudaFreeHost(houtx);
  	err = cudaFreeHost(houty);
  	err = cudaFreeHost(houtput);
  	err = cudaFreeHost(hmaskx);
  	err = cudaFreeHost(hmasky);
  	err = cudaFreeHost(hmaskx1);
  	err = cudaFreeHost(hmaskx2);

  	return 0;
}
