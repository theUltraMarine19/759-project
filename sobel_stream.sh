#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J sobel_stream
#SBATCH -o sobel_stream.out -e sobel_stream.err
#SBATCH --gres=gpu:1

module load cuda
nvcc sobel_stream.cu sobel.cu `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o sobel_stream.o
./sobel_stream.o 32 32
# cuda-memcheck ./sobel_stream.o
