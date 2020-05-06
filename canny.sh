#!/usr/bin/env bash
#SBATCH -p wacc -N 1 -c 20
#SBATCH -J canny
#SBATCH -o canny.out -e canny.err

export OMP_NUM_THREADS=20
g++ imread_canny.cpp canny.cpp sobel.cpp `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -O3  -o imread_canny.o -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
# g++ imread_canny.cpp canny.cpp sobel.cpp `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -O3 -o imread_canny.o -march=native -fopt-info-vec
# gprof imread_canny.o gmon.out > canny_omp.txt

for img in bear.jpg debug.jpg doctor.jpg gerasa.jpg hand-sanitizer.jpg landscape.jpg license.jpg Swimming-club.jpg tree.jpg whatnow.jpeg lenna.jpg
do
	./imread_canny.o $img "${img}_canny_OMP.png"
done

