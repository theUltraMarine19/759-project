#include <cmath>
#include <cstdio>
#include <iostream>
#include "canny.cuh"
using namespace std;

__global__ void generateGaussian(float *filter, float* sum, float sigma) {
	int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
	int y_idx = threadIdx.y + blockDim.y * blockDim.y;
	int sz = blockDim.x; // always odd

	volatile __shared__ float arr[]; // no otpimized loads to registers
	float* deno = &arr[0]; 									
	float* sum = &arr[(bdy + 2) * bdx];

	*deno = 2 * sigma * sigma; // memory transaction takes place immediately since volatile

	filter[y_idx*sz + x_idx] = 1.0/( exp( ( (y_idx-sz/2) * (y_idx-sz/2) + (x_idx-sz/2)*(x_idx-sz/2) )/deno ) * (deno * M_PI) );
	__syncthreads(); // wait for all threads to populate the filter values

	/* Effectively serializing the next part of code. Hurts parallelism massively */

	// Protection against all threads trying to modify this variable
	atomicAdd(sum, filter[y_idx*sz + x_idx]); // memory transaction takes place immediately since volatile
	__syncthreads(); // wiat for all threads to have updated the "sum" variable

	filter[y_idx*sz + x_idx] /= *sum;
}

__global__ void NonMaxSuppresion(float *grad, float* magn, float* supp, size_t r, size_t c) {
	
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockDim.y;
	int idx = i*c+j; // code motion

	// check for out of bounds
	if (j < c && i < r) {

		float angle = grad[idx];

		if ((-22.5 <= angle && angle <= 22.5) || (157.5 <= angle && angle <= -157.5))
			if (magn[idx] < magn[idx+1] || magn[idx] < magn[idx-1])
				supp[idx] = 0.0;

		if ((-112.5 <= angle && angle <= -67.5) || (67.5 <= angle && angle <= 112.5))
			if (magn[idx] < magn[idx+c] || magn[idx] < magn[idx-c])
				supp[idx] = 0.0;

		if ((-67.5 <= angle && angle <= -22.5) || (112.5 <= angle && angle <= 157.5))
			if (magn[idx] < magn[idx-c+1] || magn[idx] < magn[idx+c-1])
				supp[idx] = 0.0;

		if ((-157.5 <= angle && angle <= -112.5) || (22.5 <= angle && angle <= 67.5))
			if (magn[idx] < magn[idx+c+1] || magn[idx] < magn[idx-c-1])
				supp[idx] = 0.0;

	}

}

__global__ void mag_grad(float *Gx, float *Gy, float *magn, float *grad, size_t r, size_t c) {
	
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockDim.y;
	int idx = i*c+j;

	// check for out of bounds
	if (j < c && i < r) {

		magn[idx] = sqrt(Gx[idx] * Gx[idx] + Gy[idx] * Gy[idx]);
	
		// if (Gx[idx] == 0)
		// 	grad[idx] = 90;
		// else
		// 	grad[idx] = atan2(Gy[idx], Gx[idx]) * 180.0/M_PI;

		grad[idx] = (Gx[idx] == 0) * 90.0 + (Gx[idx] != 0) * (atan2(Gy[idx], Gx[idx]) * 180.0/M_PI); // Avoids thread divergence

	}

}

__global__ void hysteresis(float* supp, size_t r, size_t c, float low, float high) {
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				if (supp[i*c+j] > high) {
					supp[i*c+j] = 1.0;
					rec_hysteresis(supp, i, j, r, c, low, high);
				}
			}
		}
	
		#pragma omp for simd collapse(2)
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				if (supp[i*c+j] != 1.0) {
					supp[i*c+j] = 0.0;
				}
			}
		}
	
}

__global__ void rec_hysteresis(float *supp, size_t idxr, size_t idxc, size_t r, size_t c, float low, float high) {
	for (int i = idxr-1; i <= idxr+1; i++) {
		for (int j = idxc-1; j <= idxc+1; j++) {
			if (i < 0 || j < 0 || i >= r || j >= c)
				continue;
			if (i != idxr && j != idxc) {
				if (supp[i*c + j] != 1.0) {
					if (supp[i*c+j] > low) {
						supp[i*c+j] = 1.0;
						rec_hysteresis(supp, i, j, r, c, low, high);
					}
					else {
						supp[i*c+j] = 0.0;
					}
				}
			}
		}
	}
}