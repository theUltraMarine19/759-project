#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J canny_stream
#SBATCH -o canny_stream.out -e canny_stream.err
#SBATCH --gres=gpu:1

module load cuda
nvcc canny_stream.cu canny.cu sobel.cu `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o canny_stream.o
# nvprof --unified-memory-profiling off ./canny_stream.o 32 32 &> canny_stream.txt
# cuda-memcheck ./canny_stream.o

for img in bear.jpg debug.jpg doctor.jpg gerasa.jpg hand-sanitizer.jpg landscape.jpg license.jpg Swimming-club.jpg tree.jpg whatnow.jpeg lenna.jpg
do
	./canny_stream.o $img 32 32 "${img}_canny_stream.png"
done
