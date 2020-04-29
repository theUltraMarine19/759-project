#include <cmath>
#include <cstdio>
#include <iostream>

#include "canny.cuh"
using namespace std;

// __global__ functions can't be inlined actually
__forceinline__ __global__ void generateGaussian(float *filter, float sigma) {
	int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
	int y_idx = threadIdx.y + blockDim.y * blockIdx.y;
	int sz = blockDim.x; // always odd

	__shared__ float arr[2]; // Can't use "volatile" to prevent shmem data from being directly loaded onto registers
	// float deno = arr[0]; 									
	// float sum = arr[1];

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		arr[1] = 0;
		arr[0] = 2 * sigma * sigma; // memory transaction takes place immediately since volatile 
	}

	__syncthreads(); // all should get the sum and deno values populated

	filter[y_idx*sz + x_idx] = 1.0/( exp( ( (y_idx-sz/2) * (y_idx-sz/2) + (x_idx-sz/2)*(x_idx-sz/2) )/arr[0] ) * (arr[0] * M_PI) );
	
	/* Effectively serializing the next part of code. Hurts parallelism massively */

	// Protection against all threads trying to modify this variable
	atomicAdd(&arr[1], filter[y_idx*sz + x_idx]); // memory transaction takes place immediately since volatile
	__syncthreads(); // wiat for all threads to have updated the "sum" variable

	filter[y_idx*sz + x_idx] /= arr[1];
}

// template <int sig>
// __global__ void generateGaussian(float *filter) {

// 	float sigma = sig/100;
// 	int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
// 	int y_idx = threadIdx.y + blockDim.y * blockIdx.y;
// 	int sz = blockDim.x; // always odd

// 	__shared__ float arr[2]; // Can't use "volatile" to prevent shmem data from being directly loaded onto registers
// 	// float deno = arr[0]; 									
// 	// float sum = arr[1];

// 	if (threadIdx.x == 0 && threadIdx.y == 0) {
// 		arr[1] = 0;
// 		arr[0] = 2 * sigma * sigma; // memory transaction takes place immediately since volatile 
// 	}

// 	__syncthreads(); // all should get the sum and deno values populated

// 	filter[y_idx*sz + x_idx] = 1.0/( exp( ( (y_idx-sz/2) * (y_idx-sz/2) + (x_idx-sz/2)*(x_idx-sz/2) )/arr[0] ) * (arr[0] * M_PI) );
	
// 	/* Effectively serializing the next part of code. Hurts parallelism massively */

// 	// Protection against all threads trying to modify this variable
// 	atomicAdd(&arr[1], filter[y_idx*sz + x_idx]); // memory transaction takes place immediately since volatile
// 	__syncthreads(); // wiat for all threads to have updated the "sum" variable

// 	filter[y_idx*sz + x_idx] /= arr[1];
// }

__global__ void NonMaxSuppression(float *grad, float* magn, float* supp, size_t r, size_t c) {
	
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int bdx = blockDim.x, bdy = blockDim.y;
	int idx = i*c+j; // code motion
	int tidx = threadIdx.x, tidy = threadIdx.y;
	float avg_intensity = 0.0;

	extern __shared__ float img[]; // Can't use "volatile" to prevent shmem data from being directly loaded onto registers

	// load image elements in-place
	if (j < c && i < r)
		img[(tidy+1)*(bdx+2) + tidx+1] = magn[idx];
	else
		img[(tidy+1)*(bdx+2) + tidx+1] = avg_intensity;

	
	if (tidx == 0 && tidy == 0) { // leftmost top corner
		
		if (j >= 1 && i >= 1)
			img[tidy*(bdx+2) + tidx] = magn[idx-c-1];
		else
			img[tidy*(bdx+2) + tidx] = avg_intensity;

	}
	else if (tidx == 0 && tidy == bdy - 1) { // leftmost bottom corner
		
		if (j >= 1 && i < r-1)
			img[(tidy+2)*(bdx+2) + tidx] = magn[idx+c-1];
		else
			img[(tidy+2)*(bdx+2) + tidx] = avg_intensity;		
	
	}
	else if (tidx == bdx - 1 && tidy == 0) { // rightmost top corner
		
		if (j < c -1 && i >= 1)
			img[tidy*(bdx+2) + tidx+2] = magn[idx-c+1];
		else
			img[tidy*(bdx+2) + tidx+2] = avg_intensity;

	}
	else if (tidx == bdx - 1 && tidy == bdy -1) { // rightmost bottom corner
		
		if (j < c -1 && i < r-1)
			img[(tidy+2)*(bdx+2) + tidx+2] = magn[idx+c+1];
		else
			img[(tidy+2)*(bdx+2) + tidx+2] = avg_intensity;
	
	}


	if (tidx == 0) { // leftmost col
		
		if (j >= 1)
			img[(tidy+1)*(bdx+2) + tidx] = magn[idx-1];
		else
			img[(tidy+1)*(bdx+2) + tidx] = avg_intensity;
	
	}
	else if (tidx == bdx - 1) { // rightmost col
		
		if (j < c-1)
			img[(tidy+1)*(bdx+2) + tidx+2] = magn[idx+1];
		else
			img[(tidy+1)*(bdx+2) + tidx+2] = avg_intensity;
	
	}
	
	if (tidy == 0) { // top row
		
		if (i >= 1)
			img[tidy*(bdx+2) + tidx+1] = magn[idx-c];
		else
			img[tidy*(bdx+2) + tidx+1] = avg_intensity;
	
	}
	else if (tidy == bdy - 1) { // bottom row
	
		if (i < r-1)
			img[(tidy+2)*(bdx+2) + tidx+1] = magn[idx+c];
		else
			img[(tidy+2)*(bdx+2) + tidx+1] = avg_intensity;
	
	}

	__syncthreads();

	// check for out of bounds
	if (i > 0 && j > 0 && j < c-1 && i < r-1) {

		float angle = grad[idx];
		int idx1 = (tidy+1)*(bdx+2) + tidx+1;
		
		if ((-22.5 < angle && angle <= 22.5) || (157.5 < angle && angle <= -157.5)) {
			// printf("%f %f %f\n", img[idx1], img[idx1-1], img[idx1+1]);
			if (img[idx1] < img[idx1+1] || img[idx1] < img[idx1-1])
				supp[idx] = 0.0;
		}

		if ((-112.5 < angle && angle <= -67.5) || (67.5 < angle && angle <= 112.5)) {
			// printf("%f %f %f\n", img[idx1], img[idx1-c], img[idx1+c]);
			if (img[idx1] < img[idx1+(bdx+2)] || img[idx1] < img[idx1-(bdx+2)])
				supp[idx] = 0.0;
		}

		if ((-67.5 < angle && angle <= -22.5) || (112.5 < angle && angle <= 157.5)) {
			// printf("%f %f %f\n", img[idx1], img[idx1-c+1], img[idx1+c-1]);
			if (img[idx1] < img[idx1-(bdx+2)+1] || img[idx1] < img[idx1+(bdx+2)-1])
				supp[idx] = 0.0;
		}

		if ((-157.5 < angle && angle <= -112.5) || (22.5 < angle && angle <= 67.5)) {
			// printf("%f %f %f\n", img[idx1], img[idx1+c+1], img[idx1-c-1]);
			if (img[idx1] < img[idx1+(bdx+2)+1] || img[idx1] < img[idx1-(bdx+2)-1])
				supp[idx] = 0.0;
		}

	}

}

__global__ void mag_grad(float *Gx, float *Gy, float *magn, float *grad, size_t r, size_t c) {
	
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
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

__device__ void lock(volatile int *mutex) { // spinlock
	while (atomicCAS((int*)mutex, 0, 1) != 0);
	// other threads in the warp keep spinning, so thread in critical section can't be scheduled to release mutesx. Warp-level semantics
}

__device__ void unlock(volatile int *mutex) {
	atomicExch((int*)mutex, 0);
}


__global__ void q_init(float* supp, float high, float *q, int *back, size_t r, size_t c, int* mutex) {
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = i*c+j;

	__shared__ int arr[1];

	if (i == 0 && j == 0) {
		arr[0] = *back;
	}

	__syncthreads();

	if (i < r && j < c && supp[idx] > high) {
		supp[idx] = 1.0;

		lock(mutex);
		// push {i,j} into queue if its value > high
		q[arr[0]] = i;
		q[arr[0] + 1] = j;
		
		printf("Value of back is %d from idx %d %d\n", arr[0], i, j);
		arr[0] += 2;

		unlock(mutex);
	}
}

__global__ void hysteresis(float* supp, size_t r, size_t c, float low, float high, int* ctr) {
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = i*c+j;

	volatile __shared__ int arr[1];
	if (threadIdx.x == 0 && threadIdx.y == 0)
		arr[0] = *ctr;

	__syncthreads();

	if (i < r && j < c) {
		if (supp[idx] > high) {
			supp[idx] = 1.0;

			// unroll loops
			if (i+1 < r && j+1 < c && supp[(i+1)*c+(j+1)] > low && supp[(i+1)*c+(j+1)] != 1.0) { // southeast
				supp[(i+1)*c+(j+1)] = 1.0;
				atomicAdd((int*)&arr[0], 1);
			}
			
			if (j+1 < c && supp[i*c+(j+1)] > low && supp[i*c+(j+1)] != 1.0) {	// east
				supp[i*c+(j+1)] = 1.0;
				atomicAdd((int*)&arr[0], 1);
			}
			
			if (i+1 < r && supp[(i+1)*c+j] > low && supp[(i+1)*c+j] != 1.0) {	// south 
				supp[(i+1)*c+j] = 1.0;
				atomicAdd((int*)&arr[0], 1);
			}

			if (i-1 >= 0 && supp[(i-1)*c+j] > low && supp[(i-1)*c+j] != 1.0) { // north
				supp[(i-1)*c+j] = 1.0;
				atomicAdd((int*)&arr[0], 1);
			}

			if (j-1 >= 0 && supp[i*c+(j-1)] > low && supp[i*c+(j-1)] != 1.0) { // west
				supp[i*c+(j-1)] = 1.0;
				atomicAdd((int*)&arr[0], 1);
			}

			if (i+1 < r && j-1 >= 0 && supp[(i+1)*c+(j-1)] > low && supp[(i+1)*c+(j-1)] != 1.0) { // southwest 
				supp[(i+1)*c+(j-1)] = 1.0;
				atomicAdd((int*)&arr[0], 1);
			}

			if (i-1 >= 0 && j+1 < c && supp[(i-1)*c+(j+1)] > low && supp[(i-1)*c+(j+1)] != 1.0) { // northeast 
				supp[(i-1)*c+(j+1)] = 1.0;
				atomicAdd((int*)&arr[0], 1);
			}

			if (i-1 >= 0 && j-1 >= 0 && supp[(i-1)*c+(j-1)] > low && supp[(i-1)*c+(j-1)] != 1.0) { // northwest 
				supp[(i-1)*c+(j-1)] = 1.0;
				atomicAdd((int*)&arr[0], 1);
			}

		}
	}

	__syncthreads(); // need all other threads in warp to increment arr[0] to get correct value of *ctr
	if (threadIdx.x == 0 && threadIdx.y == 0) 
		*ctr = arr[0];
}

// template <int l, int h>
// __global__ void hysteresis(float* supp, size_t r, size_t c, int* ctr) {

// 	float low = l/100, high = h/100;
// 	int j = threadIdx.x + blockDim.x * blockIdx.x;
// 	int i = threadIdx.y + blockDim.y * blockIdx.y;
// 	int idx = i*c+j;

// 	volatile __shared__ int arr[1];
// 	if (threadIdx.x == 0 && threadIdx.y == 0)
// 		arr[0] = *ctr;

// 	__syncthreads();

// 	if (i < r && j < c) {
// 		if (supp[idx] > high) {
// 			supp[idx] = 1.0;

// 			// unroll loops
// 			if (i+1 < r && j+1 < c && supp[(i+1)*c+(j+1)] > low && supp[(i+1)*c+(j+1)] != 1.0) { // southeast
// 				supp[(i+1)*c+(j+1)] = 1.0;
// 				atomicAdd((int*)&arr[0], 1);
// 			}
			
// 			if (j+1 < c && supp[i*c+(j+1)] > low && supp[i*c+(j+1)] != 1.0) {	// east
// 				supp[i*c+(j+1)] = 1.0;
// 				atomicAdd((int*)&arr[0], 1);
// 			}
			
// 			if (i+1 < r && supp[(i+1)*c+j] > low && supp[(i+1)*c+j] != 1.0) {	// south 
// 				supp[(i+1)*c+j] = 1.0;
// 				atomicAdd((int*)&arr[0], 1);
// 			}

// 			if (i-1 >= 0 && supp[(i-1)*c+j] > low && supp[(i-1)*c+j] != 1.0) { // north
// 				supp[(i-1)*c+j] = 1.0;
// 				atomicAdd((int*)&arr[0], 1);
// 			}

// 			if (j-1 >= 0 && supp[i*c+(j-1)] > low && supp[i*c+(j-1)] != 1.0) { // west
// 				supp[i*c+(j-1)] = 1.0;
// 				atomicAdd((int*)&arr[0], 1);
// 			}

// 			if (i+1 < r && j-1 >= 0 && supp[(i+1)*c+(j-1)] > low && supp[(i+1)*c+(j-1)] != 1.0) { // southwest 
// 				supp[(i+1)*c+(j-1)] = 1.0;
// 				atomicAdd((int*)&arr[0], 1);
// 			}

// 			if (i-1 >= 0 && j+1 < c && supp[(i-1)*c+(j+1)] > low && supp[(i-1)*c+(j+1)] != 1.0) { // northeast 
// 				supp[(i-1)*c+(j+1)] = 1.0;
// 				atomicAdd((int*)&arr[0], 1);
// 			}

// 			if (i-1 >= 0 && j-1 >= 0 && supp[(i-1)*c+(j-1)] > low && supp[(i-1)*c+(j-1)] != 1.0) { // northwest 
// 				supp[(i-1)*c+(j-1)] = 1.0;
// 				atomicAdd((int*)&arr[0], 1);
// 			}

// 		}
// 	}

// 	__syncthreads(); // need all other threads in warp to increment arr[0] to get correct value of *ctr
// 	if (threadIdx.x == 0 && threadIdx.y == 0) 
// 		*ctr = arr[0];
// }
				
__global__ void weak_disconnected_edge_removal(float* supp, size_t r, size_t c) {
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = i*c+j;

	if (j < c && i < r)
		supp[idx] = (supp[idx] != 1.0) * 0.0 + (supp[idx] == 1.0) * supp[idx];

}
