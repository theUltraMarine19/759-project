#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J sobel_thrust
#SBATCH -o sobel_thrust.out -e sobel_thrust.err
#SBATCH --gres=gpu:1

module load cuda
nvcc sobel_thrust.cu sobel.cu `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o sobel_thrust.o
# nvprof --unified-memory-profiling off ./sobel_thrust.o 32 32 &> sobel_thrust.txt
# cuda-memcheck ./sobel_main.o

for img in bear.jpg debug.jpg doctor.jpg gerasa.jpg hand-sanitizer.jpg landscape.jpg license.jpg Swimming-club.jpg tree.jpg whatnow.jpeg lenna.jpg
do
	./sobel_thrust.o $img 32 32 "${img}_sobel_thrust.png"
done

