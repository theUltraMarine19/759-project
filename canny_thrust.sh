#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J canny_thrust
#SBATCH -o canny_thrust.out -e canny_thrust.err
#SBATCH --gres=gpu:1

module load cuda
nvcc canny_thrust.cu canny.cu sobel.cu `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o canny_thrust.o
# nvprof --unified-memory-profiling off ./canny_main.o 32 32 &> canny_thrust.txt
# cuda-memcheck ./canny_main.o

for img in bear.jpg debug.jpg doctor.jpg gerasa.jpg hand-sanitizer.jpg landscape.jpg license.jpg Swimming-club.jpg tree.jpg whatnow.jpeg lenna.jpg
do
	./canny_thrust.o $img 32 32 "${img}_canny_thrust.png"
done