#!/usr/bin/env bash
#SBATCH -p wacc -N 1 -c 20
#SBATCH -J sobel_s
#SBATCH -o sobel_s.out -e sobel_s.err

g++ imread.cpp sobel.cpp `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -O3 -o sobel_s.o -march=native -fopt-info-vec

for img in bear.jpg debug.jpg doctor.jpg gerasa.jpg hand-sanitizer.jpg landscape.jpg license.jpg Swimming-club.jpg tree.jpg whatnow.jpeg lenna.jpg
do
	./sobel_s.o $img 20 "${img}_sobel_serial.png"
done