#!/usr/bin/env bash
#SBATCH -p wacc -N 1 -c 20
#SBATCH -J sobel
#SBATCH -o sobel.out -e sobel.err

g++ sobel.cpp -Wall -O3 -o sobel -fopenmp
./sobel 1024 20
