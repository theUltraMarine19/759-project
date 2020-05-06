#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/core.hpp"
#include "/srv/home/arijit/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
#include <iostream>
#include <queue>
#include "canny.cuh"
#include "sobel.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform.h>

typedef thrust::tuple<float,float> Tuple2;

using namespace cv;
using namespace std;

struct gaussian {
    int sz;
    float deno;

    gaussian(int s, float sigma) { sz = s; deno = 2 * sigma * sigma; }

    __host__ __device__ 
    float operator() (int x) {
        int y_idx = x / sz;
        int x_idx = x % sz;
        return 1.0/( exp( ( (y_idx-sz/2) * (y_idx-sz/2) + (x_idx-sz/2)*(x_idx-sz/2) )/deno ) * (deno * M_PI) );
    }
};

struct fctr {
    __host__ __device__
    float operator() (float x, float sum) {
        return x/sum;
    }
};

struct magn_grad {
    __host__ __device__
    Tuple2 operator() (float x, float y) {
        float angle = (x == 0) * 90.0 + (x != 0) * (atan2(y, x) * 180.0/M_PI);
        return Tuple2(sqrt(x*x + y*y), angle);
    }
};

int main(int argc, char* argv[]) {
	
  	int bdx = atoi(argv[2]);
  	int bdy = atoi(argv[3]);
  	
  	cudaError_t err;

  	cudaEvent_t start;
  	cudaEvent_t stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);

  	Mat image, norm_image;
    image = imread(argv[1], 0); 	
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

    float *filter, *dimg, *supp, *outx, *outy, *output;
    
    err = cudaMalloc((void **)&dimg, image.rows * image.cols * sizeof(float));
    err = cudaMallocManaged((void **)&supp, image.rows * image.cols * sizeof(float));
	err = cudaMallocManaged((void **)&filter, 3 * 3 * sizeof(float));
    
    float *smooth_img, *grad;
    err = cudaMallocManaged((void **)&smooth_img, image.rows * image.cols * sizeof(float));
    err = cudaMallocManaged((void **)&grad, image.rows * image.cols * sizeof(float));

    err = cudaMemcpy(dimg, norm_image.ptr<float>(), image.rows * image.cols * sizeof(float), cudaMemcpyHostToDevice);
    
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

  	
  	int* ctr;
	cudaMallocManaged((void **)&ctr, sizeof(int));
	cudaMemset(ctr, 0, sizeof(int));  	

  	cudaEventRecord(start);

    // Just need the declarations, not the values
    dim3 block(3, 3);
    dim3 grid(1, 1);

    // generateGaussian<<<grid, block>>>(filter, 1.0);
    
    thrust::device_vector<int> init(9);
    for (int i = 0; i < 9; i++)
        init[i] = i;
    gaussian op(3, 1.0);
    thrust::transform(init.begin(), init.end(), filter, op);
    float sum = thrust::reduce(filter, filter+9, 0.0, thrust::plus<float>());
    fctr op1;
    float counts[9] = {1,2,3,4,5,6,7,8,9};
    thrust::transform(filter, filter+9, counts, filter, op1);


    conv(dimg, filter, smooth_img, image.rows, image.cols, bdx, bdy);
    err = cudaDeviceSynchronize();
    
    Mat smoothed = Mat(image.rows, image.cols, CV_32F, smooth_img);
    Mat norm_smoothed;
    normalize(smoothed, norm_smoothed, 0, 1, NORM_MINMAX, CV_32F);
    err = cudaFree(smooth_img);
    
    err = cudaMemcpy(dimg, norm_smoothed.ptr<float>(), image.rows * image.cols * sizeof(float), cudaMemcpyHostToDevice);
    
    conv_opt(dimg, dmaskx1, dmaskx2, outx, image.rows, image.cols, bdx, bdy);
    conv_opt(dimg, dmaskx2, dmaskx1, outy, image.rows, image.cols, bdx, bdy);

  	// block.x = bdx; block.y = bdy;
  	// grid.x = (image.cols + block.x - 1) / block.x; grid.y = (image.rows + block.y - 1) / block.y;
   //  mag_grad<<<grid, block>>>(outx, outy, output, grad, image.rows, image.cols);
   //  err = cudaDeviceSynchronize();

    magn_grad op2;
    thrust::transform(outx, outx + image.cols*image.rows, outy, thrust::make_zip_iterator(thrust::make_tuple(output, grad)), op2);
    
    Mat mag = Mat(image.rows, image.cols, CV_32F, output);
    Mat norm_mag;
    normalize(mag, norm_mag, 0, 1, NORM_MINMAX, CV_32F);
    
	err = cudaMemcpy(dimg, norm_mag.ptr<float>(), image.rows * image.cols * sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(supp, norm_mag.ptr<float>(), image.rows * image.cols * sizeof(float), cudaMemcpyHostToDevice);
    
	NonMaxSuppression<<<grid, block, (bdx+2)*(bdy+2)*sizeof(float)>>>(grad, dimg, supp, image.rows, image.cols);
    err = cudaDeviceSynchronize();
  	cout << cudaGetErrorName(err) << endl;

	// q_init<<<grid, block>>>(supp, 0.11, queue, back, image.rows, image.cols, mutex);
	// err = cudaDeviceSynchronize();

    // cudaEventRecord(start);

    do {

		*ctr = 0;
		hysteresis<<<grid, block>>>(supp, image.rows, image.cols, 0.08, 0.11, ctr);
        // hysteresis<8, 11><<<grid, block>>>(supp, image.rows, image.cols, ctr);
		err = cudaDeviceSynchronize();
		// cout << *ctr << endl;
	
	} while (*ctr != 0);
	
	
	weak_disconnected_edge_removal<<<grid, block>>>(supp, image.rows, image.cols);
	err = cudaDeviceSynchronize(); 
	
  	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);

  	float ms;
  	cudaEventElapsedTime(&ms, start, stop);

  	Mat out = Mat(image.rows, image.cols, CV_32F, supp);
    Mat norm_out;
    normalize(out, norm_out, 0, 1, NORM_MINMAX, CV_32F);

  	cout << ms << endl;

	Mat write_out;
	normalize(norm_out, write_out, 0, 255, NORM_MINMAX, CV_8U);
	  imwrite(argv[4], write_out);

	err = cudaFree(dimg);
  	err = cudaFree(filter);
    err = cudaFree(grad);
  	err = cudaFree(outx);
  	err = cudaFree(outy);
  	err = cudaFree(output);
    err = cudaFree(dmaskx);
  	err = cudaFree(dmasky);
  	err = cudaFree(dmaskx1);
  	err = cudaFree(dmaskx2);
  	err = cudaFree(supp);
	// err = cudaFree(mutex);
  	
  	// cout << "Done!\n";
  	
  	return 0;
}
