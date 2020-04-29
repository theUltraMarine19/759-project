#ifndef SOBEL_CUH
#define SOBEL_CUH

// Computes the convolution of image and mask, storing the result in output.
// Each thread should compute _one_ element of output.
// Shared memory allocated dynamically

// image is an array of length #rows * #cols of managed memory.
// mask is an array of length 3*3 of managed memory.
// output is an array of length #rows * #cols of managed memory.

// Assumptions:
// - 2D configuration
// - blockDim.x >= 1

// The following are stored/computed in shared memory:
// - The entire mask
// - The elements of image needed to compute the elements of output corresponding to the threads in the given block
// - The output image elements corresponding to the given block before it is written back to global memory

__global__ void conv_kernel(const float* image, const float* mask, float* output, unsigned int r, unsigned int c);

// No shared memory version

__global__ void conv_kernel_no_shmem(const float* image, const float* mask, float* output, unsigned int r, unsigned int c);

// 1D versions with shared memory

__global__ void conv_kernel_horiz(const float* image, const float* mask, float* output, unsigned int r, unsigned int c);
__global__ void conv_kernel_vert(const float* image, const float* mask, float* output, unsigned int r, unsigned int c);
__global__ void conv_kernel_vert_opt(const float* image, const float* mask, float* output, unsigned int r, unsigned int c);

// Computes the magnitude of the gradient from x-component (stored in outx) and y-component (stored in outy) into out
// Each thread computes one element of out
// No shared memory since one thread doesn't work on the data used by another thread

// outx, outy and out are 1D arrays of length #rows * #cols in managed memory 

__global__ void magnitude(const float* outx, const float *outy, float* out, unsigned int r, unsigned int c);

// Computes the convolution of image and mask, storing the result in output.
// image is an array of length #rows * #cols of managed memory.
// mask is an array of length 3*3 of managed memory.
// output is an array of length #rows * #cols of managed memory.
// Makes one call to stencil_kernel with threads_per_block threads per block.
// The kernel call is followed by a call to cudaDeviceSynchronize for timing purposes.

// Assumptions:
// - threads_per_block >= 1

__host__ void conv(const float* image, const float* mask, float* output, unsigned int r, unsigned int c, unsigned int bdx, unsigned int bdy);
__host__ void conv_opt(const float* image, const float* mask1, float* mask2, float* output, unsigned int r, unsigned int c, unsigned int bdx, unsigned int bdy);

#endif