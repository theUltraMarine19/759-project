#!/usr/bin/env bash
#SBATCH -p wacc -N 1 -c 20
#SBATCH -J sobel
#SBATCH -o sobel.out -e sobel.err

g++ imread.cpp sobel.cpp `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -O3 -o imread.o -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
# g++ imread.cpp sobel.cpp `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -O3 -o imread.o -march=native -fopt-info-vec

for img in bear.jpg debug.jpg doctor.jpg gerasa.jpg hand-sanitizer.jpg landscape.jpg license.jpg Swimming-club.jpg tree.jpg whatnow.jpeg lenna.jpg
do
	./imread.o $img 20 "${img}_sobel_OMP.png"
done