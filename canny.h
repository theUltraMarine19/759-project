#include <cstdlib>

void generateGaussian(float *filter, size_t m, float sigma);
void NonMaxSuppresion(float *grad, float* magn, float* supp, size_t r, size_t c);
void mag_grad(float *Gx, float *Gy, float *magn, float *grad, size_t r, size_t c);
// void threshold(float* supp, size_t r, size_t c, float low, float high);
// void hysteresis(float *supp, size_t r, size_t c, float low, float high);
void hysteresis(float* supp, size_t r, size_t c, float low, float high);
void rec_hysteresis(float *supp, size_t idxr, size_t idxy, size_t r, size_t c, float low, float high);