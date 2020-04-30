#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J canny_cuda
#SBATCH -o canny_cuda.out -e canny_cuda.err
#SBATCH --gres=gpu:1

module load cuda
nvcc canny_main.cu canny.cu sobel.cu `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o canny_main.o
./canny_main.o 32 32
# nvprof --unified-memory-profiling off ./canny_main.o 32 32 &> canny_cuda_new.txt
# cuda-memcheck ./canny_main.o
