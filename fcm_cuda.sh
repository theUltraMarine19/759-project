#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J fcm_cuda
#SBATCH -o fcm_cuda.out -e fcm_cuda.err
#SBATCH --gres=gpu:1

./exec_fcm_cuda.o 32 lenna.jpg

