#include <cmath>
#include <iostream>
#include <cstdio>
#include <omp.h>
#include "canny.h"
using namespace std;

omp_lock_t qlock;

void generateGaussian(float *filter, size_t m, float sigma) {
	float deno = 2 * sigma * sigma;
	float sum = 0;
	
	#pragma omp parallel 
	{	

		#pragma omp for simd reduction(+:sum) collapse(2)
		for (size_t i = 0; i < m; i++) {
			for (size_t j = 0; j < m; j++) {
				// cout << 1.0/exp(((i-m/2)*(i-m/2) + (j-m/2)*(j-m/2))/deno) << endl;
				filter[i*m+j] = 1.0/( exp(((i-m/2)*(i-m/2) + (j-m/2)*(j-m/2))/deno) * (deno * M_PI) );
				sum += filter[i*m+j];
			}
		}
	
		/* flatten to one loop in an effort for simd */
		#pragma omp for simd
		for (size_t i = 0; i < m*m; i++) {
			filter[i] /= sum;
		}
	}
}

void mag_gradient(float *Gx, float *Gy, float *magn, float *grad, size_t r, size_t c) {
	#pragma omp parallel for simd collapse(2)
	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {
			
			int idx = i*c+j;
			magn[idx] = sqrt(Gx[idx] * Gx[idx] + Gy[idx] * Gy[idx]);
			
			// might need to remove if-else for simd
			if (Gx[idx] == 0)
				grad[idx] = 90;
			else
				grad[idx] = atan2(Gy[idx], Gx[idx]) * 180.0/M_PI;
		}
	}
}

void NonMaxSuppresion(float *grad, float* magn, float* supp, size_t r, size_t c) {
	
	#pragma omp parallel for simd collapse(2)
	// ignore the pixels at border
	for (int i = 1; i < r-1; i++) {
		for (int j = 1; j < c-1; j++) {
			
			int idx = i*c+j; // code motion
			float angle = grad[idx];

			if ((-22.5 <= angle && angle <= 22.5) || (157.5 <= angle && angle <= -157.5))
				if (magn[idx] < magn[idx+1] || magn[idx] < magn[idx-1])
					supp[idx] = 0.0;

			if ((-112.5 <= angle && angle <= -67.5) || (67.5 <= angle && angle <= 112.5))
				if (magn[idx] < magn[idx+c] || magn[idx] < magn[idx-c])
					supp[idx] = 0.0;

			if ((-67.5 <= angle && angle <= -22.5) || (112.5 <= angle && angle <= 157.5))
				if (magn[idx] < magn[idx-c+1] || magn[idx] < magn[idx+c-1])
					supp[idx] = 0.0;

			if ((-157.5 <= angle && angle <= -112.5) || (22.5 <= angle && angle <= 67.5))
				if (magn[idx] < magn[idx+c+1] || magn[idx] < magn[idx-c-1])
					supp[idx] = 0.0;

		}
	}	
}

void hysteresis(float* supp, size_t r, size_t c, float low, float high) {
	#pragma omp parallel
	{
		#pragma omp for simd
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				if (supp[i*c+j] > high) {
					supp[i*c+j] = 1.0;
					rec_hysteresis(supp, i, j, r, c, low, high);
				}
			}
		}
	
		#pragma omp for simd collapse(2)
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				if (supp[i*c+j] != 1.0) {
					supp[i*c+j] = 0.0;
				}
			}
		}
	}
}

void rec_hysteresis(float *supp, size_t idxr, size_t idxc, size_t r, size_t c, float low, float high) {
	for (int i = idxr-1; i <= idxr+1; i++) {
		for (int j = idxc-1; j <= idxc+1; j++) {
			if (i < 0 || j < 0 || i >= r || j >= c)
				continue;
			if (i != idxr && j != idxc) {
				if (supp[i*c + j] != 1.0) {
					if (supp[i*c+j] > low) {
						supp[i*c+j] = 1.0;
						rec_hysteresis(supp, i, j, r, c, low, high);
					}
					// else {
					// 	supp[i*c+j] = 0.0;
					// }
				}
			}
		}
	}
}

// /* OMP Tasks Potential */
// void q_hysteresis(float* supp, size_t r, size_t c, float low, float high) {
// 	queue<pair<int, int>> q;
// 	omp_init_lock(&qlock);

// 	#pragma omp parallel
// 	{
// 		#pragma omp for collapse(2)	  // Can't have a simd with critical inside
// 		for (int i = 0; i < r; i++) {
// 			for (int j = 0; j < c; j++) {
// 				if (supp[i*c+j] > high) {
// 					supp[i*c+j] = 1.0;

// 					#pragma omp critical
// 					q.push({i, j});

// 				}
// 			}
// 		} // implicit synchronization here

// 		#pragma omp single nowait 
// 		q_rec_hysteresis(supp, q, r, c, low, high);
		
// 		#pragma omp barrier // need the synchronization at the end, so that all tasks are completed

// 		#pragma omp for simd collapse(2)
// 		for (int i = 0; i < r; i++) {
// 			for (int j = 0; j < c; j++) {
// 				// code motion not worth eating a register, let compiler do it if it wants
// 				if (supp[i*c+j] != 1.0) { 
// 					supp[i*c+j] = 0.0;
// 				}
// 			}
// 		}
// 	}
// }

// // Not doing nested parallelism on purpose
// void q_rec_hysteresis(float *supp, queue<pair<int, int>>& q, size_t r, size_t c, float low, float high) {
// 	pair<int, int> ele;
// 	long sz;
	
// 	while (true) {
// 		omp_set_lock(&qlock);
// 		sz = q.size();
// 		if (sz != 0) {
// 			ele = q.front();
// 			q.pop();
// 		}
// 		omp_unset_lock(&qlock);
// 		if (sz == 0)
// 			break;

// 		printf("Popping %d %d from %d elements by %d\n", ele.first, ele.second, q.size(), omp_get_thread_num());
// 		int idxr = ele.first, idxc = ele.second;

// 		for (int i = idxr-1; i <= idxr+1; i++) {
// 			for (int j = idxc-1; j <= idxc+1; j++) {
// 				// code motion not worth eating a register, let compiler do it if it wants
// 				if (i < 0 || j < 0 || i >= r || j >= c)
// 					continue;
// 				if (i != idxr && j != idxc) {
// 					if (supp[i*c + j] != 1.0) {
// 						if (supp[i*c+j] > low) {
// 							supp[i*c+j] = 1.0;
// 							// Only one thread, no need of critical
// 							omp_set_lock(&qlock);
// 							q.push({i, j});
// 							omp_unset_lock(&qlock);	
// 							#pragma omp task firstprivate(supp)
// 							q_rec_hysteresis(supp, q, r, c, low, high);
// 						}
// 						else {
// 							supp[i*c+j] = 0.0;
// 						}
// 					}
// 				}
// 			}
// 		}

// 	}
	
// }


/* OMP Tasks Potential */
void q_hysteresis(float* supp, size_t r, size_t c, float low, float high) {
	queue<pair<int, int>> q;
	omp_init_lock(&qlock);
	long sz;
	// int ctr = 0;

	#pragma omp parallel
	{
		#pragma omp for collapse(2) // Can't have a simd with critical inside
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				if (supp[i*c+j] > high) {
					supp[i*c+j] = 1.0;

					#pragma omp critical
					q.push({i, j});

				}
			}
		}

		#pragma omp single nowait 
		while (true) {
			omp_set_lock(&qlock);
			sz = q.size();
			omp_unset_lock(&qlock);
			// printf("size from CC : %d\n", q.size());
			if (sz == 0)
				break;
			#pragma omp task firstprivate(supp)
			q_rec_hysteresis(supp, q, r, c, low, high);
			// ctr += 1;
			
		}

		#pragma omp barrier // need the synchronization at the end, so that all tasks are completed
		// printf("%d thread is here\n", omp_get_thread_num());


		#pragma omp for simd collapse(2)
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				// code motion not worth eating a register, let compiler do it if it wants
				if (supp[i*c+j] != 1.0) { 
					supp[i*c+j] = 0.0;
				}
			}
		}
	}
}

// Not doing nested parallelism on purpose
void q_rec_hysteresis(float *supp, queue<pair<int, int>>& q, size_t r, size_t c, float low, float high) {
	
		omp_set_lock(&qlock);
		if (q.size() == 0) {
			// printf("========= ABORT ==========\n");
			omp_unset_lock(&qlock);
			return;	
		}
		auto ele = q.front();
		q.pop();
		// printf("Popping %d %d from %d elements by %d\n", ele.first, ele.second, q.size(), omp_get_thread_num());
		omp_unset_lock(&qlock);
		int idxr = ele.first, idxc = ele.second;

		for (int i = idxr-1; i <= idxr+1; i++) {
			for (int j = idxc-1; j <= idxc+1; j++) {
				// code motion not worth eating a register, let compiler do it if it wants
				if (i < 0 || j < 0 || i >= r || j >= c)
					continue;
				if (i != idxr && j != idxc) {
					if (supp[i*c + j] != 1.0) {
						if (supp[i*c+j] > low) {
							supp[i*c+j] = 1.0;
							omp_set_lock(&qlock);
							q.push({i, j});
							// printf("Pushing %d %d\n", i, j);
							omp_unset_lock(&qlock);		
						}
						else {
							supp[i*c+j] = 0.0;
						}
					}
				}
			}
		}

}

