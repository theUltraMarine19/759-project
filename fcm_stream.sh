#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J fcm_stream
#SBATCH -o fcm_stream.out -e fcm_stream.err
#SBATCH --gres=gpu:1

./fcm_stream_exec.o 32 lenna.jpg
