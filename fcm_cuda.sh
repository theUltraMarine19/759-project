#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J fcm_cuda
#SBATCH -o fcm_cuda.out -e fcm_cuda.err
#SBATCH --gres=gpu:1

./fcm_cuda lenna.jpg 32

