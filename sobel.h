#include <cstdlib>

void Convolve(const float *image, float *output, size_t r, size_t c, const float *mask, size_t m);
void convolve1D_horiz(const float* __restrict__ image, float* __restrict__ output, size_t r, size_t c, const float* __restrict__ mask, size_t m);
void convolve1D_vert(const float *image, float *output, size_t r, size_t c, const float *mask, size_t m);

void convolve1D_horiz_opt(const float* __restrict image, float* __restrict output, size_t r, size_t c, const float* __restrict mask, size_t m);
void convolve1D_vert_opt(const float* __restrict image, float* __restrict output, size_t r, size_t c, const float* __restrict mask, size_t m);