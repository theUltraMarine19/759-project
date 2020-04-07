#include <chrono>
#include <cstdlib>
#include <iostream>
#include <ratio>

#include <omp.h>

using namespace std;

void Convolve(const float *image, float *output, size_t n, const float *mask, size_t m) {
  #pragma omp parallel for collapse(2)
  // schedule(static) by default since balanced loops
  // merge nested loops into 1 (no data dependencies)
  for (size_t x = 0; x < n; x++) {
    for (size_t y = 0; y < n; y++) {
      output[x * n + y] = 0;
      // #pragma omp simd
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
          if ((x + i - (m - 1) / 2) < n && (y + j - (m - 1) / 2) < n)
            output[x * n + y] += mask[i * m + j] * image[(x + i - (m - 1) / 2) * n + (y + j - (m - 1) / 2)];
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  
  size_t n = atoi(argv[1]);
  int t = atoi(argv[2]);

  float* image = new float[n * n];
  float* outputx = new float[n * n];
  float* outputy = new float[n * n];
  
  for (size_t i = 0; i < n * n; i++) {
    image[i] = 1.0;
  }

  float maskx[9] = {-1,-2,-1,0,0,0,1,2,1};
  float masky[9] = {-1,0,1,-2,0,2,-1,0,1};

  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point end;
  chrono::duration<double, milli> duration_sec;

  omp_set_num_threads(t);
  start = chrono::high_resolution_clock::now();
  Convolve(image, outputx, n, maskx, 3);
  Convolve(image, outputy, n, masky, 3); 
  end = chrono::high_resolution_clock::now();

  duration_sec = chrono::duration_cast<chrono::duration<double, milli>>(end - start);

  // cout << output[0] << "\n"
       // << output[n * n - 1] << "\n" 
  cout << duration_sec.count() << "\n";
       

  for (size_t i = 0; i < n*n; i++)
    cout << outputx[i] << " ";
  cout << endl;

  for (size_t i = 0; i < n*n; i++)
    cout << outputy[i] << " ";
  cout << endl;

  delete[] image;
  delete[] outputx;
  delete[] outputy;
  
  return 0;
}