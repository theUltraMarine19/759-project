#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J fcm_thrust
#SBATCH -o fcm_thrust.out -e fcm_thrust.err
#SBATCH --gres=gpu:1


./fcm_thrust_exec.o 128 lenna.jpg &> t_lenna.txt
./fcm_thrust_exec.o 128 tree.jpg &> t_tree.txt
./fcm_thrust_exec.o 128 landscape.jpg &> t_land.txt
./fcm_thrust_exec.o 128 bear.jpg &> t_bear.txt
./fcm_thrust_exec.o 128 hand-sanitizer.png &> t_hand.txt
./fcm_thrust_exec.o 128 Swimming-club.jpg &> t_swim.txt
./fcm_thrust_exec.o 128 gerasa.jpg &> t_ger.txt
./fcm_thrust_exec.o 128 doctor.jpg &> t_doc.txt
./fcm_thrust_exec.o 128 whatnow.jpeg &> t_what.txt
./fcm_thrust_exec.o 1024 license.jpg &> t_license.txt
