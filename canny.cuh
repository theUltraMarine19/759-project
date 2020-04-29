#ifndef CANNY_CUH
#define CANNY_CUH

#include <cstdlib>
using namespace std;

__device__ void lock(volatile int *mutex);
__device__ void unlock(volatile int *mutex);
__global__ void generateGaussian(float *filter, float sigma);
__global__ void NonMaxSuppression(float *grad, float* magn, float* supp, size_t r, size_t c);
__global__ void mag_grad(float *Gx, float *Gy, float *magn, float *grad, size_t r, size_t c);
__global__ void q_init(float* supp, float high, float *q, int *back, size_t r, size_t c, int* mutex);
__global__ void hysteresis(float* supp, size_t r, size_t c, float low, float high, int* ctr);
__global__ void weak_disconnected_edge_removal(float* supp, size_t r, size_t c);

// template <float sigma> __global__ void generateGaussian(float *filter);
// template <float low, float high> __global__ void hysteresis(float* supp, size_t r, size_t c, int* ctr);

#endif