#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J sobel
#SBATCH -o imread.out -e imread.err

module load gcc/7.1.0
module load cuda/9
module load python/3.6.0
module load opencv

g++ -ggdb imread.cpp -o imread.o `pkg-config --cflags --libs opencv`
./imread.o