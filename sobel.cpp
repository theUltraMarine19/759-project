// #include <chrono>
#include <cstdlib>
#include <iostream>
// #include <ratio>

#include <omp.h>
#include "sobel.h"

using namespace std;

void Convolve(const float *image, float *output, size_t r, size_t c, const float *mask, size_t m) {
  #pragma omp parallel for collapse(2)
  // schedule(static) by default since balanced loops
  // merge nested loops into 1 (no data dependencies)
  for (size_t x = 0; x < r; x++) {
    for (size_t y = 0; y < c; y++) {
      output[x * c + y] = 0;
      // #pragma omp simd
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
          if ((x + i - (m - 1) / 2) < r && (y + j - (m - 1) / 2) < c) {
          	// cout << x << " " << y << " " << i << " " << j << endl;
          	output[x * c + y] += mask[i * m + j] * image[(x + i - (m - 1) / 2) * c + (y + j - (m - 1) / 2)];
          	// cout << output[x*c+y] << endl;
          }
        }
      }
    }
  }
}

void convolve1D_horiz(const float *image, float *output, size_t r, size_t c, const float *mask, size_t m) {
  #pragma omp for simd collapse(2)
  for (size_t x = 0; x < r; x++) {
    for (size_t y = 1; y < c-1; y++) {
      output[x*c+y] = mask[0] * image[x*c + (y-(m-1)/2)] + 
                      mask[1] * image[x*c + (y+1-(m-1)/2)] + 
                      mask[2] * image[x*c + (y+2-(m-1)/2)];
    }
  }
  #pragma omp for simd
  for (size_t x = 0; x < r; x++) {
    output[x*c] = mask[1] * image[x*c] + mask[2] * image[x*c + 1];
    output[x*c+c-1] = mask[0] * image[x*c + c-2] + mask[1] * image[x*c + c-1];
  }
}

void convolve1D_vert(const float *image, float *output, size_t r, size_t c, const float *mask, size_t m) {
  #pragma omp for simd collapse(2)
  for (size_t x = 1; x < r-1; x++) {
    for (size_t y = 0; y < c; y++) {
      output[x*c+y] = mask[0] * image[(x-(m-1)/2)*c + y] + 
                      mask[1] * image[(x+1-(m-1)/2)*c + y] + 
                      mask[2] * image[(x+2-(m-1)/2)*c + y];
    }
  }
  #pragma omp for simd
  for (size_t y = 0; y < c; y++) {
    output[y] = mask[1] * image[y] + mask[2] * image[c + y];
    output[(r-1)*c+y] = mask[0] * image[(r-2)*c + y] + mask[1] * image[(r-1)*c +y];
  }
}

// int main(int argc, char* argv[]) {
  
//   size_t n = atoi(argv[1]);
//   int t = atoi(argv[2]);

//   float* image = new float[n * n];

  
//   for (size_t i = 0; i < n * n; i++) {
//     image[i] = 1.0;
//   }

//   chrono::high_resolution_clock::time_point start;
//   chrono::high_resolution_clock::time_point end;
//   chrono::duration<double, milli> duration_sec;

//   omp_set_num_threads(t);
//   start = chrono::high_resolution_clock::now();
//   Convolve(image, outputx, n, maskx, 3);
//   Convolve(image, outputy, n, masky, 3); 
//   end = chrono::high_resolution_clock::now();

//   duration_sec = chrono::duration_cast<chrono::duration<double, milli>>(end - start);

//   // cout << output[0] << "\n"
//        // << output[n * n - 1] << "\n" 
//   cout << duration_sec.count() << "\n";
       

//   for (size_t i = 0; i < n*n; i++)
//     cout << outputx[i] << " ";
//   cout << endl;

//   for (size_t i = 0; i < n*n; i++)
//     cout << outputy[i] << " ";
//   cout << endl;

//   delete[] image;
//   delete[] outputx;
//   delete[] outputy;
  
//   return 0;
// }