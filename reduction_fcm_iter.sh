#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J reduction_fcm_iter
#SBATCH -o reduction_fcm_iter.out -e reduction_fcm_iter.err
#SBATCH --gres=gpu:1

./reduction_fcm_iter_exec.o 32 lenna.jpg
