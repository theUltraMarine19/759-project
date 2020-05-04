#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J reduction_fcm
#SBATCH -o reduction_fcm.out -e reduction_fcm.err
#SBATCH --gres=gpu:1

./reduction_fcm_exec.o 32 lenna.jpg
