#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J sobel_cuda
#SBATCH -o sobel_cuda.out -e sobel_cuda.err
#SBATCH --gres=gpu:1

module load cuda
nvcc sobel_main.cu sobel.cu `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o sobel_main
./sobel_main 4 4
# cuda-memcheck ./sobel_main
