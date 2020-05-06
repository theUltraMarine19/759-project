#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J sobel_cuda
#SBATCH -o sobel_cuda.out -e sobel_cuda.err
#SBATCH --gres=gpu:1

module load cuda
nvcc sobel_main.cu sobel.cu `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o sobel_main.o
# nvprof --unified-memory-profiling off ./sobel_main.o 32 32 &> sobel_cuda.txt
# nvvp ./sobel_main.o 32 32 
# cuda-memcheck ./sobel_main.o

for img in bear.jpg debug.jpg doctor.jpg gerasa.jpg hand-sanitizer.jpg landscape.jpg license.jpg Swimming-club.jpg tree.jpg whatnow.jpeg lenna.jpg
do
	./sobel_main.o $img 32 32 "${img}_sobel_cuda.png"
done
