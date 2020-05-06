#!/usr/bin/env bash
#SBATCH -p wacc -N 1 -c 20
#SBATCH -J canny_s
#SBATCH -o canny_s.out -e canny_s.err

g++ imread_canny.cpp canny.cpp sobel.cpp `pkg-config --cflags --libs ~/installation/OpenCV-3.4.4/lib64/pkgconfig/opencv.pc` -O3 -o canny_s.o -march=native -fopt-info-vec
# gprof canny_s.o gmon.out > canny_serial.txt

for img in bear.jpg debug.jpg doctor.jpg gerasa.jpg hand-sanitizer.jpg landscape.jpg license.jpg Swimming-club.jpg tree.jpg whatnow.jpeg lenna.jpg
do
	./canny_s.o $img "${img}_canny_serial.png"
done

