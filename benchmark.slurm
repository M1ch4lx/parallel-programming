#!/bin/bash -l

#SBATCH -A plgar2026-cpu
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:20:00
#SBATCH -J membrane
#SBATCH -o benchmark-%j.out
#SBATCH -e benchmark-%j.err

module load scipy-bundle/2021.10-intel-2021b

export SLURM_OVERLAP=1

cd "$SLURM_SUBMIT_DIR"

mkdir -p results

python benchmark.py --sizes 1025,2049 --processes 1,2,4,8,16,32 --repeats 3 --max-iters 150000 --output results/benchmark.csv

python plot_benchmarks.py --input results/benchmark.csv --output-dir results/plots