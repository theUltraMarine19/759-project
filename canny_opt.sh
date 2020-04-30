#!/usr/bin/env bash
#SBATCH -p wacc -N 2 -c 20
#SBATCH -J canny_opt
#SBATCH -o canny_opt.out -e canny_opt.err
#SBATCH --gres=gpu:1

export OMP_PLACES=threads
export OMP_PROC_BIND=spread
module load cuda
nvcc canny_opt.cu canny.cu sobel.cu canny.cpp `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -Xcompiler -fopenmp -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o canny_opt.o
./canny_opt.o 32 32 20
# nvprof --unified-memory-profiling off ./canny_opt.o 32 32 40 &> canny_opt.txt
# cuda-memcheck ./canny_main.o
