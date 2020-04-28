#include <cmath>
#include <cstdio>
#include <iostream>
#include "sobel.cuh"
using namespace std;

__global__ void conv_kernel_no_shmem(const float* image, const float* mask, float* output, unsigned int r, unsigned int c) {

	int tidx = threadIdx.x, tidy = threadIdx.y;
	int bidx = blockIdx.x, bidy = blockIdx.y;
	int bdy = blockDim.y, bdx = blockDim.x;
	float avg_intensity = 0;

	long x_idx = tidx + (long)bdx * (long)bidx; 		// long since can be > 2^31 -1
	long y_idx = tidy + (long)bdy * (long)bidy;

	int idx = y_idx*c+x_idx;
	
	float temp = 0;
	
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			
			if (x_idx-1+j < c && y_idx-1+i < r && x_idx-1+j >= 0 && y_idx-1+i >= 0)
				temp += image[(y_idx-1+i)*c + x_idx-1+j] * mask[i*3+j];
			else	
				temp += avg_intensity * mask[i*3+j];
		}
	}

	if (y_idx < r && x_idx < c)
		output[idx] = temp;

}

__global__ void conv_kernel(const float* image, const float* mask, float* output, unsigned int r, unsigned int c) {
	
	int tidx = threadIdx.x, tidy = threadIdx.y;
	int bidx = blockIdx.x, bidy = blockIdx.y;
	int bdy = blockDim.y, bdx = blockDim.x;
	float avg_intensity = 0;

	// printf("%d %d %d %d\n", tidx, tidy, bidx, bidy);
	// if (tidx == 0 && tidy == 0 && bidx == 0 && bidy == 0) {
		
		// printf("%d %d\n", bidx, bidy);
		// for (int i = 0; i < 9; i++)
		// 	printf("%f ", mask[i]);
		// printf("\n");
	// }
	

	extern __shared__ float arr[]; // Can't use "volatile" to prevent shmem data from being directly loaded onto registers
	float* img = &arr[0]; 									
	float* msk = &arr[(bdx + 2) * (bdy + 2)]; 
	float* out = &arr[(bdx + 2) * (bdy + 2) + 3*3]; 		 

	long x_idx = tidx + (long)bdx * (long)bidx; 		// long since can be > 2^31 -1
	long y_idx = tidy + (long)bdy * (long)bidy;
	
	// load image elements in-place
	if (x_idx < c && y_idx < r)
		img[(tidy+1)*(bdx+2) + tidx+1] = image[y_idx * c + x_idx];
	else
		img[(tidy+1)*(bdx+2) + tidx+1] = avg_intensity;

	
	if (tidx < 3 && tidy < 3)
		msk[tidy*3 + tidx] = mask[tidy*3 + tidx];

	
	if (tidx == 0 && tidy == 0) { // leftmost top corner
		
		if (x_idx >= 1 && y_idx >= 1)
			img[tidy*(bdx+2) + tidx] = image[(y_idx-1) * c + x_idx-1];
		else
			img[tidy*(bdx+2) + tidx] = avg_intensity;

	}
	else if (tidx == 0 && tidy == bdy - 1) { // leftmost bottom corner
		
		if (x_idx >= 1 && y_idx < r-1)
			img[(tidy+2)*(bdx+2) + tidx] = image[(y_idx+1) * c + x_idx-1];
		else
			img[(tidy+2)*(bdx+2) + tidx] = avg_intensity;		
	
	}
	else if (tidx == bdx - 1 && tidy == 0) { // rightmost top corner
		
		if (x_idx < c -1 && y_idx >= 1)
			img[tidy*(bdx+2) + tidx+2] = image[(y_idx-1) * c + x_idx+1];
		else
			img[tidy*(bdx+2) + tidx+2] = avg_intensity;

	}
	else if (tidx == bdx - 1 && tidy == bdy -1) { // rightmost bottom corner
		
		if (x_idx < c -1 && y_idx < r-1)
			img[(tidy+2)*(bdx+2) + tidx+2] = image[(y_idx+1) * c + x_idx+1];
		else
			img[(tidy+2)*(bdx+2) + tidx+2] = avg_intensity;
	
	}


	if (tidx == 0) { // leftmost col
		
		if (x_idx >= 1)
			img[(tidy+1)*(bdx+2) + tidx] = image[y_idx*c + x_idx-1];
		else
			img[(tidy+1)*(bdx+2) + tidx] = avg_intensity;
	
	}
	else if (tidx == bdx - 1) { // rightmost col
		
		if (x_idx < c-1)
			img[(tidy+1)*(bdx+2) + tidx+2] = image[y_idx*c + x_idx+1];
		else
			img[(tidy+1)*(bdx+2) + tidx+2] = avg_intensity;
	
	}
	
	if (tidy == 0) { // top row
		
		if (y_idx >= 1)
			img[tidy*(bdx+2) + tidx+1] = image[(y_idx-1)*c + x_idx];
		else
			img[tidy*(bdx+2) + tidx+1] = avg_intensity;
	
	}
	else if (tidy == bdy - 1) { // bottom row
	
		if (y_idx < r-1)
			img[(tidy+2)*(bdx+2) + tidx+1] = image[(y_idx+1)*c + x_idx];
		else
			img[(tidy+2)*(bdx+2) + tidx+1] = avg_intensity;
	
	}

	__syncthreads();

	// if (tidx == 2 && tidy == 1 && bidx == 21 && bidy == 30) {
		
	// 	for (int i = 0; i < bdy+2; i++) {
	// 		for (int j = 0; j < bdx+2; j++ ) {
	// 			printf("%f ", img[i*(bdx+2)+j]);	
	// 		}
	// 		printf("\n");	
	// 	}

	// 	for (int i = -1; i < bdy+1; i++) {
	// 		for (int j = -1; j < bdx+1; j++ ) {
	// 			printf("%f ", image[(bidy*bdy+i)*c+(bidx*bdx+j)]);	
	// 		}
	// 		printf("\n");	
	// 	}		
	// }

	out[tidy*bdx+tidx] = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			out[tidy*bdx+tidx] += img[(tidy+i)*(bdx+2) + (tidx+j)] * msk[i*3+j];	
		}		
	}

	__syncthreads();

	if (x_idx < c && y_idx < r)
		output[y_idx*c+x_idx] = out[tidy*bdy+tidx];

}

__global__ void conv_kernel_horiz(const float* image, const float* mask, float* output, unsigned int r, unsigned int c) {
	int tidx = threadIdx.x, tidy = threadIdx.y;
	int bidx = blockIdx.x, bidy = blockIdx.y;
	int bdy = blockDim.y, bdx = blockDim.x;
	float avg_intensity = 0;

	extern __shared__ float arr[];		// Can't use "volatile" to prevent shmem data from being directly loaded onto registers
	float* img = &arr[0]; 									
	float* msk = &arr[(bdx + 2) * bdy]; 
	float* out = &arr[(bdx + 2) * bdy + 3]; 		 

	long x_idx = tidx + (long)bdx * (long)bidx; 		// long since can be > 2^31 -1
	long y_idx = tidy + (long)bdy * (long)bidy;

	// load image elements in-place
	if (x_idx < c && y_idx < r)
		img[tidy*(bdx+2) + tidx+1] = image[y_idx * c + x_idx];
	else
		img[tidy*(bdx+2) + tidx+1] = avg_intensity;

	// load the mask into shmem
	if (tidx < 3 && tidy == 0)
		msk[tidx] = mask[tidx];

	if (tidx == 0) { // leftmost col
		
		if (x_idx >= 1)
			img[tidy*(bdx+2) + tidx] = image[y_idx*c + x_idx-1];
		else
			img[tidy*(bdx+2) + tidx] = avg_intensity;
	
	}
	else if (tidx == bdx - 1) { // rightmost col
		
		if (x_idx < c-1)
			img[tidy*(bdx+2) + tidx+2] = image[y_idx*c + x_idx+1];
		else
			img[tidy*(bdx+2) + tidx+2] = avg_intensity;
	
	}

	__syncthreads();

	out[tidy*bdx+tidx] = 0;
	for (int i = 0; i < 3; i++) {
		out[tidy*bdx+tidx] += img[tidy*(bdx+2) + (tidx+i)] * msk[i];	
	}	

	__syncthreads();

	if (x_idx < c && y_idx < r)
		output[y_idx*c+x_idx] = out[tidy*bdy+tidx];

}

__global__ void conv_kernel_vert(const float* image, const float* mask, float* output, unsigned int r, unsigned int c) {
	int tidx = threadIdx.x, tidy = threadIdx.y;
	int bidx = blockIdx.x, bidy = blockIdx.y;
	int bdy = blockDim.y, bdx = blockDim.x;
	float avg_intensity = 0;

	extern __shared__ float arr[];		// Can't use "volatile" to prevent shmem data from being directly loaded onto registers
	float* img = &arr[0]; 									
	float* msk = &arr[(bdy + 2) * bdx]; 
	float* out = &arr[(bdy + 2) * bdx + 3]; 		 

	long x_idx = tidx + (long)bdx * (long)bidx; 		// long since can be > 2^31 -1
	long y_idx = tidy + (long)bdy * (long)bidy;

	// load image elements in-place
	if (x_idx < c && y_idx < r)
		img[(tidy+1)*bdx + tidx] = image[y_idx * c + x_idx];
	else
		img[(tidy+1)*bdx + tidx] = avg_intensity;

	// load the mask into shmem
	if (tidx == 0 && tidx < 3)
		msk[tidy] = mask[tidy];

	if (tidy == 0) { // top row
		
		if (y_idx >= 1)
			img[tidy*bdx + tidx] = image[(y_idx-1)*c + x_idx];
		else
			img[tidy*bdx + tidx] = avg_intensity;
	
	}
	else if (tidy == bdy - 1) { // bottom row
	
		if (y_idx < r-1)
			img[(tidy+2)*bdx + tidx] = image[(y_idx+1)*c + x_idx];
		else
			img[(tidy+2)*bdx + tidx] = avg_intensity;
	
	}

	__syncthreads();

	out[tidy*bdx+tidx] = 0;
	for (int i = 0; i < 3; i++) {
		out[tidy*bdx+tidx] += img[(tidy+i)*bdx + tidx] * msk[i];	
	}	

	__syncthreads();

	if (x_idx < c && y_idx < r)
		output[y_idx*c+x_idx] = out[tidy*bdy+tidx];

}

__host__ void conv(const float* image, const float* mask, float* output, unsigned int r, unsigned int c, unsigned int bdx, unsigned int bdy) {

	dim3 block(bdx, bdy);
	dim3 grid((c + block.x - 1) / block.x, (r + block.y - 1) / block.y);
	// cout << bdx << " " << bdy << " " << (c + block.x - 1) / block.x << " " << (r + block.y - 1) / block.y << endl;
	// for (int i = 0; i < r; i++) {
	// 	for (int j = 0; j < c; j++) {
	// 		cout << image[i*c+j] << " ";
	// 	}
	// 	cout << endl;
	// }

	conv_kernel<<<grid, block, sizeof(float) * (bdx + 2) * (bdy + 2) + 3 * 3 * sizeof(float) + sizeof(float) * bdx * bdy>>>(image, mask, output, r, c);
	// conv_kernel_no_shmem<<<grid, block>>>(image, mask, output, r, c);
	
	cudaError_t err;
	// // Check for kernel launch errors
	// err = cudaGetLastError();
	// if (err != cudaSuccess) 
 	// 	  printf("Error: %s\n", cudaGetErrorString(err));
  	

  	err = cudaDeviceSynchronize();
  	cout << cudaGetErrorName(err) << endl;
  	
}

__host__ void conv_opt(const float* image, const float* mask1, float* mask2, float* output, unsigned int r, unsigned int c, unsigned int bdx, unsigned int bdy) {

	dim3 block(bdx, bdy);
	dim3 grid((c + block.x - 1) / block.x, (r + block.y - 1) / block.y);

	cudaError_t err;
	
	float *temp;
	err = cudaMalloc((void **)&temp, r * c * sizeof(float));	
	// cout << cudaGetErrorName(err) << endl;
	
	conv_kernel_horiz<<<grid, block, sizeof(float) * bdy * (bdx+2) + 3 * sizeof(float) + sizeof(float) * bdx * bdy>>>(image, mask1, temp, r, c);
	err = cudaDeviceSynchronize();
  	cout << cudaGetErrorName(err) << endl;
	
	conv_kernel_vert<<<grid, block, sizeof(float) * (bdy+2) * bdx + 3 * sizeof(float) + sizeof(float) * bdx * bdy>>>(temp, mask2, output, r, c);
	err = cudaDeviceSynchronize();
  	cout << cudaGetErrorName(err) << endl;

  	err = cudaFree(temp);
  	// cout << cudaGetErrorName(err) << endl;
  	
}

__global__ void magnitude(const float* outx, const float *outy, float* out, unsigned int r, unsigned int c) {
	int x_idx = threadIdx.x + (long)blockDim.x * (long)blockIdx.x;
	int y_idx = threadIdx.y + (long)blockDim.y * (long)blockIdx.y;
	int idx = y_idx*c + x_idx; // Code motion

	if (x_idx < c && y_idx < r)
		out[idx] = sqrt(outx[idx]*outx[idx] + outy[idx]*outy[idx]);
}
