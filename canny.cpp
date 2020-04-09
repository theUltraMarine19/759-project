#include <cstddef>
#include <cmath>

void generateGaussian(float *filter[], size_t m, float sigma) {
	float deno = 2 * sigma * sigma;
	float sum = 0;
	#pragma omp for collapse(2) reduction(+:sum)
	for (size_t i = -m/2; i <= m/2; i++) {
		for (size_t j = -m/2; j <= m/2; j++) {
			filter[i+m/2][j+m/2] = exp(-(i*i+j*j)/deno) * 1/(deno * M_PI);
			sum += filter[i+m/2][j+m/2];
		}
	}

	#pragma omp for collapse(2)
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < m; j++) {
			filter[i][j] /= sum;
		}
	}
}

// Do a convolution of image with the Gaussian filter

// get the output Gx, by convolution with Sobel operator

// get the output Gy, by convolution with Sobel operator

void mag_grad(float *Gx[], float *Gy[], float *magn[], float *grad[], size_t n) {
	#pragma omp for collapse(2)
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			// need to ensure between 0 and 255
			magn[i][j] = sqrt(Gx[i][j] * Gx[i][j] + Gy[i][j] * Gy[i][j]);
			if (Gx[i][j] == 0)
				grad[i][j] = 90;
			else
				grad[i][j] = atan2(Gy[i][j], Gx[i][j]);
		}
	}
}

void NonMaxSuppresion(float *grad[], float* magn[], float* supp[], size_t n) {
	#pragma omp for collapse(2)
	// ignore the pixels at border
	for (int i = 1; i < n-1; i++) {
		for (int j = 1; j < n-1; j++) {
			float angle = grad[i][j];
			if ((-22.5 <= angle && angle <= 22.5) || (157.5 <= angle && angle <= -157.5))
				if (magn[i][j] < magn[i][j+1] || magn[i][j] < magn[i][j-1])
					supp[i][j] = 0.0;

			if ((-112.5 <= angle && angle <= -67.5) || (67.5 <= angle && angle <= 112.5))
				if (magn[i][j] < magn[i+1][j] || magn[i][j] < magn[i-1][j])
					supp[i][j] = 0.0;

			if ((-67.5 <= angle && angle <= -22.5) || (112.5 <= angle && angle <= 157.5))
				if (magn[i][j] < magn[i-1][j+1] || magn[i][j] < magn[i+1][j-1])
					supp[i][j] = 0.0;

			if ((-157.5 <= angle && angle <= -112.5) || (22.5 <= angle && angle <= 67.5))
				if (magn[i][j] < magn[i+1][j+1] || magn[i][j] < magn[i-1][j-1])
					supp[i][j] = 0.0;

		}
	}

	#pragma omp for
	for (int i = 0; i < n; i++) {
		supp[i][0] = grad[i][0];
		supp[i][n-1] = grad[i][n-1];
	}
	#pragma omp for
	for (int j = 0; j < n; i++) {
		supp[0][j] = grad[0][j];
		supp[n-1][j] = grad[n-1][j];
	}	
}

void threshold(float* supp[], size_t n, float low, float high) {
	#pragma omp for collapse(2)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (supp[i][j] < low)
				supp[i][j] = 0.0;
			if (supp[i][j] > high)
				supp[i][j] = 255.0;
		}
	}
}

// pixels between low and high thresholds
// void hysteresis(float *supp[], size_t n) {
// 	#pragma omp for collapse(2)
// 	for (int i = 0; i < n; i++) {
// 		for (int j = 0; j < n; j++) {
// 			bool hasHighNei, hasMidNei;
// 			for (size_t r = i-1; r <= i+1; r++) {
// 				for (size_t c = j-1; c <= j+1; c++) {
					
// 				}
// 			}
// 		}
// 	}
// }
