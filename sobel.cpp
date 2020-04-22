#include <cstdlib>
#include <iostream>

#include "sobel.h"

using namespace std;

void Convolve(const float *image, float *output, size_t r, size_t c, const float *mask, size_t m) {
  #pragma omp for collapse(2)
  // schedule(static) by default since balanced loops
  // merge nested loops into 1 (no data dependencies)
  for (size_t x = 0; x < r; x++) {
    for (size_t y = 0; y < c; y++) {
      
      output[x * c + y] = 0;
      
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
          if ((x + i - (m - 1) / 2) < r && (y + j - (m - 1) / 2) < c) {
          	
            output[x * c + y] += mask[i * m + j] * (float)image[(x + i - (m - 1) / 2) * c + (y + j - (m - 1) / 2)];
          
          }
        }
      }
    }
  }
}

void convolve1D_horiz(const float* __restrict image, float* __restrict output, size_t r, size_t c, const float* __restrict mask, size_t m) {
  
      #pragma omp for simd collapse(2) nowait
      for (size_t x = 0; x < r; x++) {
        for (size_t y = 1; y < c-1; y++) {
          output[(x*c+y)*1] = mask[0] * image[x*c + (y-(m-1)/2)] + 
                          mask[1] * image[x*c + (y+1-(m-1)/2)] + 
                          mask[2] * image[x*c + (y+2-(m-1)/2)];
        }
      }


      #pragma omp for simd
      for (size_t x = 0; x < r; x++) {
        output[(x*c)*1] = mask[1] * image[x*c] + mask[2] * image[x*c + 1];
        output[(x*c+c-1)*1] = mask[0] * image[x*c + c-2] + mask[1] * image[x*c + c-1];
      }
}

// Store the intermediate output in column major format for next convolution. Loop interchange will cause loss in locality (1 LHS access to 3 RHS accesses) 
void convolve1D_horiz_opt(const float* __restrict image, float* __restrict output, size_t r, size_t c, const float* __restrict mask, size_t m) {
  
      #pragma omp for simd collapse(2) nowait
      for (size_t x = 0; x < r; x++) {
          for (size_t y = 1; y < c-1; y++) {
            output[x+y*r] = mask[0] * image[x*c + (y-(m-1)/2)] + 
                            mask[1] * image[x*c + (y+1-(m-1)/2)] + 
                            mask[2] * image[x*c + (y+2-(m-1)/2)];
          }
      }
  
      #pragma omp for simd
      for (size_t x = 0; x < r; x++) {
          output[x] = mask[1] * image[x*c] + mask[2] * image[x*c + 1];
          output[x+r*(c-1)] = mask[0] * image[x*c + c-2] + mask[1] * image[x*c + c-1];
      }
  
}


void convolve1D_vert_opt(const float* __restrict image, float* __restrict output, size_t r, size_t c, const float* __restrict mask, size_t m) {
    
    
        #pragma omp for simd collapse(2) nowait
        for (size_t y = 0; y < c; y++) {
          for (size_t x = 1; x < r-1; x++) {
            output[x*c+y] = mask[0] * image[(x-(m-1)/2) + y*r] + 
                            mask[1] * image[(x+1-(m-1)/2) + y*r] + 
                            mask[2] * image[(x+2-(m-1)/2) + y*r];
          }
        }
    
        #pragma omp for simd
        for (size_t y = 0; y < c; y++) {
          output[y] = mask[1] * image[y*r] + mask[2] * image[1 + y*r];
          output[(r-1)*c+y] = mask[0] * image[(r-2) + y*r] + mask[1] * image[(r-1) +y*r];
        }
    

}

void convolve1D_vert(const float* __restrict image, float* __restrict output, size_t r, size_t c, const float* __restrict mask, size_t m) {
  
    
        #pragma omp for simd collapse(2) nowait
        for (size_t x = 1; x < r-1; x++) {
          for (size_t y = 0; y < c; y++) {
            // This hurts spatial locality
            output[(x*c+y)*1] = mask[0] * image[(x-(m-1)/2)*c + y] + 
                            mask[1] * image[(x+1-(m-1)/2)*c + y] + 
                            mask[2] * image[(x+2-(m-1)/2)*c + y];
          }
        }
  
        #pragma omp for simd
        for (size_t y = 0; y < c; y++) {
          output[y*1] = mask[1] * image[y] + mask[2] * image[c + y];
          output[((r-1)*c+y)*1] = mask[0] * image[(r-2)*c + y] + mask[1] * image[(r-1)*c +y];
        }
    
}

