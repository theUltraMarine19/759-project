#include <cmath>
#include <iostream>
#include "canny.h"

using namespace std;

void generateGaussian(float *filter, size_t m, float sigma) {
	float deno = 2 * sigma * sigma;
	float sum = 0;
	#pragma omp for collapse(2) simd reduction(+:sum)
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < m; j++) {
			// cout << 1.0/exp(((i-m/2)*(i-m/2) + (j-m/2)*(j-m/2))/deno) << endl;
			filter[i*m+j] = 1.0/( exp(((i-m/2)*(i-m/2) + (j-m/2)*(j-m/2))/deno) * (deno * M_PI) );
			sum += filter[i*m+j];
		}
	}
	cout << "Midpt\n";
	#pragma omp for simd collapse(2)
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < m; j++) {
			filter[i*m + j] /= sum;
		}
	}
}

void mag_grad(float *Gx, float *Gy, float *magn, float *grad, size_t r, size_t c) {
	#pragma omp for simd collapse(2)
	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			// need to ensure between 0 and 255
			magn[i*c+j] = sqrt(Gx[i*c+j] * Gx[i*c+j] + Gy[i*c+j] * Gy[i*c+j]);
			// might need to remove if-else for simd
			if (Gx[i*c+j] == 0)
				grad[i*c+j] = 90;
			else
				grad[i*c+j] = atan2(Gy[i*c+j], Gx[i*c+j]);
		}
	}
}

void NonMaxSuppresion(float *grad, float* magn, float* supp, size_t r, size_t c) {
	#pragma omp for collapse(2)
	// ignore the pixels at border
	for (int i = 1; i < r-1; i++) {
		for (int j = 1; j < c-1; j++) {
			float angle = grad[i*c+j];
			if ((-22.5 <= angle && angle <= 22.5) || (157.5 <= angle && angle <= -157.5))
				if (magn[i*c+j] < magn[i*c+j+1] || magn[i*c+j] < magn[i*c+j-1])
					supp[i*c+j] = 0.0;

			if ((-112.5 <= angle && angle <= -67.5) || (67.5 <= angle && angle <= 112.5))
				if (magn[i*c+j] < magn[(i+1)*c+j] || magn[i*c+j] < magn[(i-1)*c+j])
					supp[i*c+j] = 0.0;

			if ((-67.5 <= angle && angle <= -22.5) || (112.5 <= angle && angle <= 157.5))
				if (magn[i*c+j] < magn[(i-1)*c+j+1] || magn[i*c+j] < magn[(i+1)*c+j-1])
					supp[i*c+j] = 0.0;

			if ((-157.5 <= angle && angle <= -112.5) || (22.5 <= angle && angle <= 67.5))
				if (magn[i*c+j] < magn[(i+1)*c+j+1] || magn[i*c+j] < magn[(i-1)*c+j-1])
					supp[i*c+j] = 0.0;

		}
	}

	#pragma omp for
	for (int i = 0; i < r; i++) {
		supp[i*c+0] = grad[i*c+0];
		supp[i*c+c-1] = grad[i*c+c-1];
	}
	#pragma omp for
	for (int j = 0; j < c; j++) {
		supp[0*c+j] = grad[0*c+j];
		supp[(r-1)*c+j] = grad[(r-1)*c+j];
	}	
}

void threshold(float* supp, size_t r, size_t c, float low, float high) {
	#pragma omp for collapse(2)
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			if (supp[i*c+j] < low)
				supp[i*c+j] = 0.0;
			if (supp[i*c+j] > high)
				supp[i*c+j] = 255.0;
		}
	}
}

// pixels between low and high thresholds
void hysteresis(float *supp, size_t r, size_t c, float low, float high) {
	#pragma omp for collapse(2)
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			if (supp[r*c+c] >= low && supp[r*c+c] <= high) {
				bool hasHighNei = false, hasMidNei = false;
				for (size_t pr = i-1; pr <= i+1; pr++) {
					for (size_t pc = j-1; pc <= j+1; pc++) {
						if (pr < 0 || pc < 0 || pr >= r || pc >= c)
							continue;
						else {
							if (supp[pr*c+pc] > high) {
								hasHighNei = true;
								supp[i*c+j] = 255.0;
								break;
							}
							else if (supp[pr*c+pc] > low && supp[pr*c+pc] < high)
								hasMidNei = true;
						}
					}
					if (hasHighNei)
						break;
				}

			}
			// due to this pixel now being classified as edge, other previous pixels need to be reconsidered
			// so hasMidNei
		}
	}
}
