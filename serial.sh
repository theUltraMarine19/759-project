#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J serial
#SBATCH -o serial.out -e serial.err
#SBATCH --gres=gpu:1

./exec_fcm.o  lenna.jpg &> s_lenna.txt
./exec_fcm.o  tree.jpg &> s_tree.txt
./exec_fcm.o  landscape.jpg &> s_land.txt
./exec_fcm.o  bear.jpg &> s_bear.txt
./exec_fcm.o  hand-sanitizer.png &> s_hand.txt
./exec_fcm.o  Swimming-club.jpg &> s_swim.txt
./exec_fcm.o  gerasa.jpg &> s_ger.txt
./exec_fcm.o  doctor.jpg &> s_doc.txt
./exec_fcm.o  whatnow.jpeg &> s_what.txt
./exec_fcm.o  license.jpg &> s_license.txt