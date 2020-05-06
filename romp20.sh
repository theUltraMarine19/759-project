#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J romp20
#SBATCH -o romp20.out -e romp20.err
#SBATCH --nodes=1 --cpus-per-task=20


./exec_fcm_omp.o  lenna.jpg &> o_lenna.txt
./exec_fcm_omp.o  tree.jpg &> o_tree.txt
./exec_fcm_omp.o  landscape.jpg &> o_land.txt
./exec_fcm_omp.o  bear.jpg &> o_bear.txt
./exec_fcm_omp.o  hand-sanitizer.png &> o_hand.txt
./exec_fcm_omp.o  Swimming-club.jpg &> o_swim.txt
./exec_fcm_omp.o  gerasa.jpg &> o_ger.txt
./exec_fcm_omp.o  doctor.jpg &> o_doc.txt
./exec_fcm_omp.o  whatnow.jpeg &> o_what.txt
./exec_fcm_omp.o  license.jpg &> o_license.txt