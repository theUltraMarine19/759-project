#include <cstdlib>
#include <queue>
using namespace std;

void generateGaussian(float *filter, size_t m, float sigma);
void NonMaxSuppresion(float *grad, float* magn, float* supp, size_t r, size_t c);
void mag_gradient(float *Gx, float *Gy, float *magn, float *grad, size_t r, size_t c);
void q_hysteresis(float* supp, size_t r, size_t c, float low, float high);
void q_rec_hysteresis(float *supp, queue<pair<int, int>>& q, size_t r, size_t c, float low, float high);
void hysteresis(float* supp, size_t r, size_t c, float low, float high);
void rec_hysteresis(float *supp, size_t idxr, size_t idxy, size_t r, size_t c, float low, float high);