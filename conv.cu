#include <cstdio>
#include "stencil.cuh"
using namespace std;

__global__ void conv_kernel(const float* image, const float* mask, float* output, unsigned int r, unsigned int c) {
	
	int tidx = threadIdx.x, tidy = threadIdx.y;
	int bidx = blockIdx.x, bidy = blockIdx.y;
	int bdy = blockDim.y, bdx = blockDim.x;
	float avg_intensity = 0.5;

	extern __shared__ float arr[];
	float* img = &arr[0]; 									
	float* msk = &arr[(bdx + 2) * (bdy + 2)]; 
	float* out = &arr[(bdx + 2) * (bdy + 2) + 3*3]; 		 

	long x_idx = tidx + (long)bdx * (long)bidx; 		// long since can be > 2^31 -1
	long y_idx = tidy + (long)bdy * (long)bidy;
	
	// load image elements in-lace
	if (x_idx < c && y_idx < r)
		img[(tidy+1)*bdy + tidx+1] = image[y_idx * c + x_idx];
	else
		img[(tidy+1)*bdy + tidx+1] = avg_intensity;

	if (tidx < 3 && tidy < 3)
		msk[tidy*bdy + tidx] = mask[tidy*bdy + tidx];

	if (tidx == 0 && tidy == 0) { // leftmost top corner
		
		if (x_idx >= 1 && y_idx >= 1)
			img[tidy*bdy + tidx] = image[(y_idx-1) * c + x_idx-1];
		else
			img[tidy*bdy + tidx] = avg_intensity;

	}
	else if (tidx == 0 && tidy == bdy - 1) { // leftmost bottom corner
		if (x_idx >= 1 && y_idx < bdy-1)
			img[(tidy+2)*bdy + tidx] = image[(y_idx+1) * c + x_idx-1];
		else
			img[(tidy+2)*bdy + tidx] = avg_intensity;		
	}
	else if (tidx == bdx - 1 && tidy == 0) { // rightmost top corner
		if (x_idx < bdx -1 && y_idx >= 1)
			img[tidy*bdy + tidx+2] = image[(y_idx-1) * c + x_idx+1];
		else
			img[tidy*bdy + tidx+2] = avg_intensity;

	}
	else if (tidx == bdx - 1 && tidy == bdy -1) { // rightmost bottom corner
		if (x_idx < bdx -1 && y_idx < bdy-1)
			img[(tidy+2)*bdy + tidx+2] = image[(y_idx+1) * c + x_idx+1];
		else
			img[(tidy+2)*bdy + tidx+2] = avg_intensity;
	}


	if (tidx == 0) { // leftmost col
		if (x_idx >= 1)
			img[(tidy+1)*bdy + tidx] = image[y_idx*c + x_idx-1];
		else
			img[(tidy+1)*bdy + tidx] = avg_intensity;
	}
	else if (tidx == bdx - 1) { // rightmost col
		if (x_idx < bdx-1)
			img[(tidy+1)*bdy + tidx+2] = image[y_idx*c + x_idx+1];
		else
			img[(tidy+1)*bdy + tidx+2] = avg_intensity;
	}
	else if (tidy == 0) { // top row
		if (y_idx >= 1)
			img[tidy*bdy + tidx+1] = image[(y_idx-1)*c + x_idx];
		else
			img[tidy*bdy + tidx+1] = avg_intensity;
	}
	else if (tidy == bdy - 1) { // bottom row
		if (y_idx < bdy-1)
			img[(tidy+2)*bdy + tidx+1] = image[(y_idx+1)*c + x_idx];
		else
			img[(tidy+2)*bdy + tidx+1] = avg_intensity;
	}

	__syncthreads();

	out[tidx] = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			out[tidy*bdy+tidx] += img[(tidy+i)*bdy + (tidx+j)] * msk[i*3+j];	
		}		
	}

	__syncthreads();

	if (x_idx < c && y_idx < r)
		output[y_idx*c+r] = out[tidy*bdy+tidx];

}

__host__ void conv(const float* image, const float* mask, float* output, unsigned int r, unsigned int c, unsigned int bdx, unsigned int bdy) {

	dim3 block(bdx, bdy);
	dim3 grid((c + block.x - 1) / block.x, (r + block.y - 1) / block.y);
  	conv_kernel<<<grid, block, sizeof(float) * (bdx + 2) * (bdy + 2) + 3 * 3 * sizeof(float) + sizeof(float) * bdx * bdy>>>(image, mask, output, r, c);
  	cudaDeviceSynchronize();
}
