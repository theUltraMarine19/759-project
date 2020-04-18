#include <cstdlib>

void Convolve(const float *image, float *output, size_t r, size_t c, const float *mask, size_t m);
void convolve1D_horiz(const float *image, float *output, size_t r, size_t c, const float *mask, size_t m);
void convolve1D_vert(const float *image, float *output, size_t r, size_t c, const float *mask, size_t m);