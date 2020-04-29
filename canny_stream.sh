#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J canny_stream
#SBATCH -o canny_stream.out -e canny_stream.err
#SBATCH --gres=gpu:1

module load cuda
nvcc canny_stream.cu canny.cu sobel.cu `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o canny_stream.o
./canny_stream.o 32 32
# cuda-memcheck ./canny_stream.o
