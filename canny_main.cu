#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/core.hpp"
#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
#include <iostream>
#include <queue>
#include "canny.cuh"
#include "sobel.cuh"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
	
  	int bdx = atoi(argv[1]);
  	int bdy = atoi(argv[2]);
  	
  	cudaError_t err;

  	cudaEvent_t start;
  	cudaEvent_t stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);

  	Mat image, norm_image;
    image = imread("license.jpg", 0); 	
    if(image.empty())                   
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    
    normalize(image, norm_image, 0, 1, NORM_MINMAX, CV_32F);
    float *img = norm_image.ptr<float>(0);

    cout << image.rows << " " << image.cols << endl;

    float maskx[9] = {-1,-2,-1,0,0,0,1,2,1};
    float masky[9] = {-1,0,1,-2,0,2,-1,0,1};

    float maskx1[3] = {1, 0, -1};
    float maskx2[3] = {1, 2, 1};

    // float masky1[3] = {1, 2, 1};
    // float masky2[3] = {1, 0, -1};

    float *filter, *sigma, *dimg, *outx, *outy, *output;
    
    err = cudaMalloc((void **)&dimg, image.rows * image.cols * sizeof(float));
	err = cudaMallocManaged((void **)&filter, 3 * 3 * sizeof(float));
    err = cudaMallocManaged((void **)&sigma, sizeof(float));

    float *smooth_img, *grad;
    err = cudaMallocManaged((void **)&smooth_img, image.rows * image.cols * sizeof(float));
    err = cudaMallocManaged((void **)&grad, image.rows * image.cols * sizeof(float));

    err = cudaMemcpy(dimg, norm_image.ptr<float>(), image.rows * image.cols * sizeof(float), cudaMemcpyHostToDevice);
    // cout << cudaGetErrorName(err) << endl;

    err = cudaMallocManaged((void **)&outx, image.rows * image.cols * sizeof(float));
  	err = cudaMallocManaged((void **)&outy, image.rows * image.cols * sizeof(float));
  	err = cudaMallocManaged((void **)&output, image.rows * image.cols * sizeof(float));
  	
  	float *dmaskx, *dmasky, *dmaskx1, *dmaskx2; // masky1 = maskx2 and masky2 = maskx1
  	err = cudaMalloc((void **)&dmaskx, 9 * sizeof(float));
  	err = cudaMalloc((void **)&dmasky, 9 * sizeof(float));
  	err = cudaMalloc((void **)&dmaskx1, 3 * sizeof(float));
  	err = cudaMalloc((void **)&dmaskx2, 3 * sizeof(float));

  	err = cudaMemcpy(dmaskx, maskx, 9 * sizeof(float), cudaMemcpyHostToDevice);
  	err = cudaMemcpy(dmasky, masky, 9 * sizeof(float), cudaMemcpyHostToDevice);
  	err = cudaMemcpy(dmaskx1, maskx1, 3 * sizeof(float), cudaMemcpyHostToDevice);
  	err = cudaMemcpy(dmaskx2, maskx2, 3 * sizeof(float), cudaMemcpyHostToDevice);
  	
  	// Can be improved with CUDA streams
  	cudaEventRecord(start);

    *sigma = 1.0;
    dim3 block(3, 3);
    dim3 grid(1, 1);
    generateGaussian<<<grid, block>>>(filter, sigma);
    err = cudaDeviceSynchronize();
    cout << cudaGetErrorName(err) << endl;

    // for (int i = 0; i < 9; i++)
    // 	cout << filter[i] << " ";
    // cout << endl;

    conv(dimg, filter, smooth_img, image.rows, image.cols, bdx, bdy);
    err = cudaDeviceSynchronize();
    // cout << cudaGetErrorName(err) << endl;
    
    Mat smoothed = Mat(image.rows, image.cols, CV_32F, smooth_img);
    Mat norm_smoothed;
    normalize(smoothed, norm_smoothed, 0, 1, NORM_MINMAX, CV_32F);
    err = cudaFree(smooth_img);
    // cout << cudaGetErrorName(err) << endl;
    
    smooth_img = norm_smoothed.ptr<float>(0);

    err = cudaMemcpy(dimg, norm_smoothed.ptr<float>(), image.rows * image.cols * sizeof(float), cudaMemcpyHostToDevice);
    cout << cudaGetErrorName(err) << endl;

    conv_opt(dimg, dmaskx1, dmaskx2, outx, image.rows, image.cols, bdx, bdy);
    conv_opt(dimg, dmaskx2, dmaskx1, outy, image.rows, image.cols, bdx, bdy);

  	block.x = bdx; block.y = bdy;
  	grid.x = (image.cols + block.x - 1) / block.x; grid.y = (image.rows + block.y - 1) / block.y;
    mag_grad<<<grid, block>>>(outx, outy, output, grad, image.rows, image.cols);
    err = cudaDeviceSynchronize();
  	// cout << cudaGetErrorName(err) << endl;
  	
  	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);

  	float ms;
  	cudaEventElapsedTime(&ms, start, stop);

  	Mat out = Mat(image.rows, image.cols, CV_32F, output);
    Mat norm_out;
    normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_32F);

  	// cout << ms << endl;

	Mat write_out;
	normalize(norm_out, write_out, 0, 255, NORM_MINMAX, CV_8U);
	imwrite("canny1_CUDA.png", write_out);

	err = cudaFree(dimg);
  	err = cudaFree(filter);
    err = cudaFree(sigma);
    err = cudaFree(grad);
  	err = cudaFree(outx);
  	err = cudaFree(outy);
  	err = cudaFree(output);
  	err = cudaFree(dmaskx);
  	err = cudaFree(dmasky);
  	err = cudaFree(dmaskx1);
  	err = cudaFree(dmaskx2);
  	
  	cout << "Done!\n";
  	
  	return 0;
}
