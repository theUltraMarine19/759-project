#ifndef CANNY_CUH
#define CANNY_CUH

#include <cstdlib>
using namespace std;

__global__ void generateGaussian(float *filter, size_t m, float sigma);
__global__ void NonMaxSuppresion(float *grad, float* magn, float* supp, size_t r, size_t c);
__global__ void mag_grad(float *Gx, float *Gy, float *magn, float *grad, size_t r, size_t c);
__global__ void hysteresis(float* supp, size_t r, size_t c, float low, float high);
__global__ void rec_hysteresis(float *supp, size_t idxr, size_t idxy, size_t r, size_t c, float low, float high);

#endif