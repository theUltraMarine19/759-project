#!/usr/bin/env bash
#SBATCH -p wacc -N 1 -c 20
#SBATCH -J canny
#SBATCH -o canny.out -e canny.err

# g++ imread_canny.cpp canny.cpp sobel.cpp `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -O3 -o imread_canny.o -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
g++ imread_canny.cpp canny.cpp sobel.cpp `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -O3 -o imread_canny.o -march=native -fopt-info-vec
./imread_canny.o 20
