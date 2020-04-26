// Author: Nic Olsen

#ifndef CONV_CUH
#define CONV_CUH

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

// Computes the convolution of image and mask, storing the result in output.
// image is an array of length #rows * #cols of managed memory.
// mask is an array of length 3*3 of managed memory.
// output is an array of length #rows * #cols of managed memory.
// Makes one call to stencil_kernel with threads_per_block threads per block.
// The kernel call is followed by a call to cudaDeviceSynchronize for timing purposes.

// Assumptions:
// - threads_per_block >= 1

__host__ void conv(const float* image, const float* mask, float* output, unsigned int r, unsigned int c, unsigned int bdx, unsigned int bdy);

#endif